from __future__ import annotations
import math
from collections import defaultdict
import scipy.constants as const
from pymatgen.core import Composition, Element, Species
class BatteryAnalyzer:
    """A suite of methods for starting with an oxidized structure and determining its potential as a battery."""

    def __init__(self, struct_oxid, working_ion='Li', oxi_override=None):
        """Pass in a structure for analysis.

        Arguments:
            struct_oxid: a Structure object; oxidation states *must* be assigned for this structure; disordered
                structures should be OK
            working_ion: a String symbol or Element for the working ion.
            oxi_override: a dict of String element symbol, Integer oxidation state pairs.
                by default, H, C, N, O, F, S, Cl, Se, Br, Te, I are considered anions.
        """
        for site in struct_oxid:
            if not hasattr(site.specie, 'oxi_state'):
                raise ValueError('BatteryAnalyzer requires oxidation states assigned to structure!')
        self.struct_oxid = struct_oxid
        self.oxi_override = oxi_override or {}
        self.comp = self.struct_oxid.composition
        if not isinstance(working_ion, Element):
            self.working_ion = Element(working_ion)
        if self.working_ion.symbol in self.oxi_override:
            self.working_ion_charge = self.oxi_override[self.working_ion.symbol]
        elif self.working_ion.symbol in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Se', 'Br', 'Te', 'I']:
            self.working_ion_charge = self.working_ion.min_oxidation_state
        else:
            self.working_ion_charge = self.working_ion.max_oxidation_state

    @property
    def max_ion_removal(self):
        """Maximum number of ion A that can be removed while maintaining charge-balance.

        Returns:
            integer amount of ion. Depends on cell size (this is an 'extrinsic' function!)
        """
        if self.working_ion_charge < 0:
            lowest_oxid = defaultdict(lambda: 2, {'Cu': 1})
            pot_sum = sum(((min((os for os in Element(spec.symbol).oxidation_states if os >= lowest_oxid[spec.symbol])) - spec.oxi_state) * self.comp[spec] for spec in self.comp if is_redox_active_intercalation(Element(spec.symbol))))
        else:
            pot_sum = sum(((Element(spec.symbol).max_oxidation_state - spec.oxi_state) * self.comp[spec] for spec in self.comp if is_redox_active_intercalation(Element(spec.symbol))))
        redox_limit = pot_sum / self.working_ion_charge
        num_working_ion = self.comp[Species(self.working_ion.symbol, self.working_ion_charge)]
        return min(redox_limit, num_working_ion)

    @property
    def max_ion_insertion(self):
        """Maximum number of ion A that can be inserted while maintaining charge-balance.
        No consideration is given to whether there (geometrically speaking) are ion sites to actually accommodate the
        extra ions.

        Returns:
            integer amount of ion. Depends on cell size (this is an 'extrinsic' function!)
        """
        if self.working_ion_charge < 0:
            pot_sum = sum(((spec.oxi_state - Element(spec.symbol).max_oxidation_state) * self.comp[spec] for spec in self.comp if is_redox_active_intercalation(Element(spec.symbol))))
        else:
            lowest_oxid = defaultdict(lambda: 2, {'Cu': 1})
            pot_sum = sum(((spec.oxi_state - min((os for os in Element(spec.symbol).oxidation_states if os >= lowest_oxid[spec.symbol]))) * self.comp[spec] for spec in self.comp if is_redox_active_intercalation(Element(spec.symbol))))
        return pot_sum / self.working_ion_charge

    def _get_max_cap_ah(self, remove, insert):
        """Give max capacity in mAh for inserting and removing a charged ion
        This method does not normalize the capacity and intended as a helper method.
        """
        num_working_ions = 0
        if remove:
            num_working_ions += self.max_ion_removal
        if insert:
            num_working_ions += self.max_ion_insertion
        return num_working_ions * abs(self.working_ion_charge) * ELECTRON_TO_AMPERE_HOURS

    def get_max_capgrav(self, remove=True, insert=True):
        """Give max capacity in mAh/g for inserting and removing a charged ion
        Note that the weight is normalized to the most ion-packed state,
        thus removal of 1 Li from LiFePO4 gives the same capacity as insertion of 1 Li into FePO4.

        Args:
            remove: (bool) whether to allow ion removal
            insert: (bool) whether to allow ion insertion

        Returns:
            max grav capacity in mAh/g
        """
        weight = self.comp.weight
        if insert:
            weight += self.max_ion_insertion * self.working_ion.atomic_mass
        return self._get_max_cap_ah(remove, insert) / (weight / 1000)

    def get_max_capvol(self, remove=True, insert=True, volume=None):
        """Give max capacity in mAh/cc for inserting and removing a charged ion into base structure.

        Args:
            remove: (bool) whether to allow ion removal
            insert: (bool) whether to allow ion insertion
            volume: (float) volume to use for normalization (default=volume of initial structure)

        Returns:
            max vol capacity in mAh/cc
        """
        vol = volume or self.struct_oxid.volume
        return self._get_max_cap_ah(remove, insert) * 1000 * 1e+24 / (vol * const.N_A)

    def get_removals_int_oxid(self):
        """Returns a set of ion removal steps, e.g. set([1 2 4]) etc. in order to
        produce integer oxidation states of the redox metals.
        If multiple redox metals are present, all combinations of reduction/oxidation are tested.
        Note that having more than 3 redox metals will likely slow down the algorithm.

        Examples:
            LiFePO4 will return [1]
            Li4Fe3Mn1(PO4)4 will return [1, 2, 3, 4])
            Li6V4(PO4)6 will return [4, 6])  *note that this example is not normalized*

        Returns:
            array of integer ion removals. If you double the unit cell, your answers will be twice as large!
        """
        oxid_els = [Element(spec.symbol) for spec in self.comp if is_redox_active_intercalation(spec)]
        num_a = set()
        for oxid_el in oxid_els:
            num_a = num_a | self._get_int_removals_helper(self.comp.copy(), oxid_el, oxid_els, num_a)
        num_working_ion = self.comp[Species(self.working_ion.symbol, self.working_ion_charge)]
        return {num_working_ion - a for a in num_a}

    def _get_int_removals_helper(self, spec_amts_oxi, redox_el, redox_els, num_a):
        """This is a helper method for get_removals_int_oxid!

        Args:
            spec_amts_oxi: a dict of species to their amounts in the structure
            redox_el: the element to oxidize or reduce
            redox_els: the full list of elements that might be oxidized or reduced
            num_a: a running set of numbers of A ion at integer oxidation steps

        Returns:
            a set of numbers A; steps for oxidizing oxid_el first, then the other oxid_els in this list
        """
        if self.working_ion_charge < 0:
            oxid_old = max((spec.oxi_state for spec in spec_amts_oxi if spec.symbol == redox_el.symbol))
            oxid_new = math.ceil(oxid_old - 1)
            lowest_oxid = defaultdict(lambda: 2, {'Cu': 1})
            if oxid_new < min((os for os in Element(redox_el.symbol).oxidation_states if os >= lowest_oxid[redox_el.symbol])):
                return num_a
        else:
            oxid_old = min((spec.oxi_state for spec in spec_amts_oxi if spec.symbol == redox_el.symbol))
            oxid_new = math.floor(oxid_old + 1)
            if oxid_new > redox_el.max_oxidation_state:
                return num_a
        spec_old = Species(redox_el.symbol, oxid_old)
        spec_new = Species(redox_el.symbol, oxid_new)
        spec_amt = spec_amts_oxi[spec_old]
        spec_amts_oxi = {sp: amt for sp, amt in spec_amts_oxi.items() if sp != spec_old}
        spec_amts_oxi[spec_new] = spec_amt
        spec_amts_oxi = Composition(spec_amts_oxi)
        oxi_noA = sum((spec.oxi_state * spec_amts_oxi[spec] for spec in spec_amts_oxi if spec.symbol not in self.working_ion.symbol))
        a = max(0, -oxi_noA / self.working_ion_charge)
        num_a = num_a | {a}
        if a == 0:
            return num_a
        for red in redox_els:
            num_a = num_a | self._get_int_removals_helper(spec_amts_oxi.copy(), red, redox_els, num_a)
        return num_a