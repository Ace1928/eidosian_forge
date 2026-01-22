from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.feff import Header, Potential, Tags
class LDos(MSONable):
    """Parser for ldos files ldos01, ldos02, ....."""

    def __init__(self, complete_dos, charge_transfer):
        """
        Args:
            complete_dos (CompleteDos): complete dos object
            charge_transfer (dict): computed charge transfer between atoms
                dictionary.
        """
        self.complete_dos = complete_dos
        self.charge_transfer = charge_transfer

    @classmethod
    def from_file(cls, feff_inp_file: str='feff.inp', ldos_file: str='ldos') -> Self:
        """
        Creates LDos object from raw Feff ldos files by
        by assuming they are numbered consecutively, i.e. ldos01.dat
        ldos02.dat...

        Args:
            feff_inp_file (str): input file of run to obtain structure
            ldos_file (str): output ldos file of run to obtain dos info, etc.
        """
        header_str = Header.header_string_from_file(feff_inp_file)
        header = Header.from_str(header_str)
        structure = header.struct
        n_sites = len(structure)
        parameters = Tags.from_file(feff_inp_file)
        if 'RECIPROCAL' in parameters:
            pot_dict = {}
            pot_readstart = re.compile('.*iz.*lmaxsc.*xnatph.*xion.*folp.*')
            pot_readend = re.compile('.*ExternalPot.*switch.*')
            pot_inp = re.sub('feff.inp', 'pot.inp', feff_inp_file)
            dos_index = 1
            begin = 0
            with zopen(pot_inp, mode='r') as potfile:
                for line in potfile:
                    if len(pot_readend.findall(line)) > 0:
                        break
                    if begin == 1:
                        begin += 1
                        continue
                    if begin == 2:
                        z_number = int(line.strip().split()[0])
                        ele_name = Element.from_Z(z_number).name
                        if ele_name not in pot_dict:
                            pot_dict[ele_name] = dos_index
                        else:
                            pot_dict[ele_name] = min(dos_index, pot_dict[ele_name])
                        dos_index += 1
                    if len(pot_readstart.findall(line)) > 0:
                        begin = 1
        else:
            pot_string = Potential.pot_string_from_file(feff_inp_file)
            dicts = Potential.pot_dict_from_str(pot_string)
            pot_dict = dicts[0]
        with zopen(ldos_file + '00.dat', mode='r') as file:
            lines = file.readlines()
        efermi = float(lines[0].split()[4])
        dos_energies = []
        ldos = {}
        for idx in range(1, len(pot_dict) + 1):
            if len(str(idx)) == 1:
                ldos[idx] = np.loadtxt(f'{ldos_file}0{idx}.dat')
            else:
                ldos[idx] = np.loadtxt(f'{ldos_file}{idx}.dat')
        for idx in range(len(ldos[1])):
            dos_energies.append(ldos[1][idx][0])
        all_pdos: list[dict] = []
        vorb = {'s': Orbital.s, 'p': Orbital.py, 'd': Orbital.dxy, 'f': Orbital.f0}
        forb = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        d_length = len(ldos[1])
        for idx in range(n_sites):
            pot_index = pot_dict[structure.species[idx].symbol]
            all_pdos.append(defaultdict(dict))
            for k, v in vorb.items():
                density = [ldos[pot_index][j][forb[k] + 1] for j in range(d_length)]
                updos = density
                downdos = None
                if downdos:
                    all_pdos[-1][v] = {Spin.up: updos, Spin.down: downdos}
                else:
                    all_pdos[-1][v] = {Spin.up: updos}
        pdos = all_pdos
        vorb2 = {0: Orbital.s, 1: Orbital.py, 2: Orbital.dxy, 3: Orbital.f0}
        pdoss = {structure[i]: {v: pdos[i][v] for v in vorb2.values()} for i in range(len(pdos))}
        forb = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        t_dos = [0] * d_length
        for idx in range(n_sites):
            pot_index = pot_dict[structure.species[idx].symbol]
            for forb_val in forb.values():
                density = [ldos[pot_index][j][forb_val + 1] for j in range(d_length)]
                for j in range(d_length):
                    t_dos[j] = t_dos[j] + density[j]
        _t_dos: dict = {Spin.up: t_dos}
        dos = Dos(efermi, dos_energies, _t_dos)
        complete_dos = CompleteDos(structure, dos, pdoss)
        charge_transfer = LDos.charge_transfer_from_file(feff_inp_file, ldos_file)
        return cls(complete_dos, charge_transfer)

    @staticmethod
    def charge_transfer_from_file(feff_inp_file, ldos_file):
        """
        Get charge transfer from file.

        Args:
            feff_inp_file (str): name of feff.inp file for run
            ldos_file (str): ldos filename for run, assume consecutive order,
                i.e., ldos01.dat, ldos02.dat....

        Returns:
            dictionary of dictionaries in order of potential sites
            ({"p": 0.154, "s": 0.078, "d": 0.0, "tot": 0.232}, ...)
        """
        cht = {}
        parameters = Tags.from_file(feff_inp_file)
        if 'RECIPROCAL' in parameters:
            dicts = [{}]
            pot_dict = {}
            dos_index = 1
            begin = 0
            pot_inp = re.sub('feff.inp', 'pot.inp', feff_inp_file)
            pot_readstart = re.compile('.*iz.*lmaxsc.*xnatph.*xion.*folp.*')
            pot_readend = re.compile('.*ExternalPot.*switch.*')
            with zopen(pot_inp, mode='r') as potfile:
                for line in potfile:
                    if len(pot_readend.findall(line)) > 0:
                        break
                    if begin == 1:
                        z_number = int(line.strip().split()[0])
                        ele_name = Element.from_Z(z_number).name
                        if len(pot_dict) == 0:
                            pot_dict[0] = ele_name
                        elif len(pot_dict) > 0:
                            pot_dict[max(pot_dict) + 1] = ele_name
                        begin += 1
                        continue
                    if begin == 2:
                        z_number = int(line.strip().split()[0])
                        ele_name = Element.from_Z(z_number).name
                        dicts[0][ele_name] = dos_index
                        dos_index += 1
                        if len(pot_dict) == 0:
                            pot_dict[0] = ele_name
                        elif len(pot_dict) > 0:
                            pot_dict[max(pot_dict) + 1] = ele_name
                    if len(pot_readstart.findall(line)) > 0:
                        begin = 1
        else:
            pot_string = Potential.pot_string_from_file(feff_inp_file)
            dicts = Potential.pot_dict_from_str(pot_string)
            pot_dict = dicts[1]
        for idx in range(len(dicts[0]) + 1):
            if len(str(idx)) == 1:
                with zopen(f'{ldos_file}0{idx}.dat', mode='rt') as file:
                    lines = file.readlines()
                    s = float(lines[3].split()[2])
                    p = float(lines[4].split()[2])
                    d = float(lines[5].split()[2])
                    f1 = float(lines[6].split()[2])
                    tot = float(lines[1].split()[4])
                    cht[str(idx)] = {pot_dict[idx]: {'s': s, 'p': p, 'd': d, 'f': f1, 'tot': tot}}
            else:
                with zopen(f'{ldos_file}{idx}.dat', mode='rt') as file:
                    lines = file.readlines()
                    s = float(lines[3].split()[2])
                    p = float(lines[4].split()[2])
                    d = float(lines[5].split()[2])
                    f1 = float(lines[6].split()[2])
                    tot = float(lines[1].split()[4])
                    cht[str(idx)] = {pot_dict[idx]: {'s': s, 'p': p, 'd': d, 'f': f1, 'tot': tot}}
        return cht

    def charge_transfer_to_str(self):
        """Returns charge transfer as string."""
        ch = self.charge_transfer
        chts = ['\nCharge Transfer\n\nabsorbing atom']
        for i in range(len(ch)):
            for atom, v2 in ch[str(i)].items():
                a = ['\n', atom, '\n', 's   ', str(v2['s']), '\n', 'p   ', str(v2['p']), '\n', 'd   ', str(v2['d']), '\n', 'f   ', str(v2['f']), '\n', 'tot ', str(v2['tot']), '\n']
                chts.extend(a)
        return ''.join(chts)