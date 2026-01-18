from collections.abc import Mapping, Sequence
from subprocess import check_call, DEVNULL
from os import unlink
from pathlib import Path
import numpy as np
from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
def write_pov(self, path):
    """Write pov file."""
    point_lights = '\n'.join((f'light_source {{{pa(loc)} {pc(rgb)}}}' for loc, rgb in self.point_lights))
    area_light = ''
    if self.area_light is not None:
        loc, color, width, height, nx, ny = self.area_light
        area_light += f'\nlight_source {{{pa(loc)} {pc(color)}\n  area_light <{width:.2f}, 0, 0>, <0, {height:.2f}, 0>, {nx:n}, {ny:n}\n  adaptive 1 jitter}}'
    fog = ''
    if self.depth_cueing and self.cue_density >= 0.0001:
        if self.cue_density > 10000.0:
            dist = 0.0001
        else:
            dist = 1.0 / self.cue_density
        fog += f'fog {{fog_type 1 distance {dist:.4f} color {pc(self.background)}}}'
    mat_style_keys = (f'#declare {k} = {v}' for k, v in self.material_styles_dict.items())
    mat_style_keys = '\n'.join(mat_style_keys)
    cell_vertices = ''
    if self.cell_vertices is not None:
        for c in range(3):
            for j in ([0, 0], [1, 0], [1, 1], [0, 1]):
                p1 = self.cell_vertices[tuple(j[:c]) + (0,) + tuple(j[c:])]
                p2 = self.cell_vertices[tuple(j[:c]) + (1,) + tuple(j[c:])]
                distance = np.linalg.norm(p2 - p1)
                if distance < 1e-12:
                    continue
                cell_vertices += f'cylinder {{{pa(p1)}, {pa(p2)}, Rcell pigment {{Black}}}}\n'
        cell_vertices = cell_vertices.strip('\n')
    a = 0
    atoms = ''
    for loc, dia, col in zip(self.positions, self.diameters, self.colors):
        tex = 'ase3'
        trans = 0.0
        if self.textures is not None:
            tex = self.textures[a]
        if self.transmittances is not None:
            trans = self.transmittances[a]
        atoms += f'atom({pa(loc)}, {dia / 2.0:.2f}, {pc(col)}, {trans}, {tex}) // #{a:n}\n'
        a += 1
    atoms = atoms.strip('\n')
    bondatoms = ''
    for pair in self.bondatoms:
        if len(pair) == 2:
            a, b = pair
            offset = (0, 0, 0)
            bond_order = 1
            bond_offset = (0, 0, 0)
        elif len(pair) == 3:
            a, b, offset = pair
            bond_order = 1
            bond_offset = (0, 0, 0)
        elif len(pair) == 4:
            a, b, offset, bond_order = pair
            bond_offset = (self.bondlinewidth, self.bondlinewidth, 0)
        elif len(pair) > 4:
            a, b, offset, bond_order, bond_offset = pair
        else:
            raise RuntimeError('Each list in bondatom must have at least 2 entries. Error at %s' % pair)
        if len(offset) != 3:
            raise ValueError('offset must have 3 elements. Error at %s' % pair)
        if len(bond_offset) != 3:
            raise ValueError('bond_offset must have 3 elements. Error at %s' % pair)
        if bond_order not in [0, 1, 2, 3]:
            raise ValueError('bond_order must be either 0, 1, 2, or 3. Error at %s' % pair)
        if bond_order > 1 and np.linalg.norm(bond_offset) > 1e-09:
            tmp_atoms = Atoms('H3')
            tmp_atoms.set_cell(self.cell)
            tmp_atoms.set_positions([self.positions[a], self.positions[b], self.positions[b] + np.array(bond_offset)])
            tmp_atoms.center()
            tmp_atoms.set_angle(0, 1, 2, 90)
            bond_offset = tmp_atoms[2].position - tmp_atoms[1].position
        R = np.dot(offset, self.cell)
        mida = 0.5 * (self.positions[a] + self.positions[b] + R)
        midb = 0.5 * (self.positions[a] + self.positions[b] - R)
        if self.textures is not None:
            texa = self.textures[a]
            texb = self.textures[b]
        else:
            texa = texb = 'ase3'
        if self.transmittances is not None:
            transa = self.transmittances[a]
            transb = self.transmittances[b]
        else:
            transa = transb = 0.0
        posa = self.positions[a]
        posb = self.positions[b]
        cola = self.colors[a]
        colb = self.colors[b]
        if bond_order == 1:
            draw_tuples = ((posa, mida, cola, transa, texa), (posb, midb, colb, transb, texb))
        elif bond_order == 2:
            bs = [x / 2 for x in bond_offset]
            draw_tuples = ((posa - bs, mida - bs, cola, transa, texa), (posb - bs, midb - bs, colb, transb, texb), (posa + bs, mida + bs, cola, transa, texa), (posb + bs, midb + bs, colb, transb, texb))
        elif bond_order == 3:
            bs = bond_offset
            draw_tuples = ((posa, mida, cola, transa, texa), (posb, midb, colb, transb, texb), (posa + bs, mida + bs, cola, transa, texa), (posb + bs, midb + bs, colb, transb, texb), (posa - bs, mida - bs, cola, transa, texa), (posb - bs, midb - bs, colb, transb, texb))
        bondatoms += ''.join((f'cylinder {{{pa(p)}, {pa(m)}, Rbond texture{{pigment {{color {pc(c)} transmit {tr}}} finish{{{tx}}}}}}}\n' for p, m, c, tr, tx in draw_tuples))
    bondatoms = bondatoms.strip('\n')
    constraints = ''
    if self.exportconstraints:
        for a in self.constrainatoms:
            dia = self.diameters[a]
            loc = self.positions[a]
            trans = 0.0
            if self.transmittances is not None:
                trans = self.transmittances[a]
            constraints += f'constrain({pa(loc)}, {dia / 2.0:.2f}, Black, {trans}, {tex}) // #{a:n} \n'
    constraints = constraints.strip('\n')
    pov = f'#include "colors.inc"\n#include "finish.inc"\n\nglobal_settings {{assumed_gamma 1 max_trace_level 6}}\nbackground {{{pc(self.background)}{(' transmit 1.0' if self.transparent else '')}}}\ncamera {{{self.camera_type}\n  right -{self.image_width:.2f}*x up {self.image_height:.2f}*y\n  direction {self.image_plane:.2f}*z\n  location <0,0,{self.camera_dist:.2f}> look_at <0,0,0>}}\n{point_lights}\n{(area_light if area_light != '' else '// no area light')}\n{(fog if fog != '' else '// no fog')}\n{mat_style_keys}\n#declare Rcell = {self.celllinewidth:.3f};\n#declare Rbond = {self.bondlinewidth:.3f};\n\n#macro atom(LOC, R, COL, TRANS, FIN)\n  sphere{{LOC, R texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}\n#end\n#macro constrain(LOC, R, COL, TRANS FIN)\nunion{{torus{{R, Rcell rotate 45*z texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}\n     torus{{R, Rcell rotate -45*z texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}\n     translate LOC}}\n#end\n\n{(cell_vertices if cell_vertices != '' else '// no cell vertices')}\n{atoms}\n{bondatoms}\n{(constraints if constraints != '' else '// no constraints')}\n'
    with open(path, 'w') as fd:
        fd.write(pov)
    return path