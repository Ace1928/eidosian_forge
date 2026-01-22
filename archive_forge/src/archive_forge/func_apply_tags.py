from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
def apply_tags(self):
    """Apply tags."""
    tags = {}
    for tag in self.tags:
        istruct = tag.get('istruct', 'all')
        if istruct not in ('all', self.istruct):
            continue
        site_index = tag['site_index']
        color = tag.get('color', [0.5, 0.5, 0.5])
        opacity = tag.get('opacity', 0.5)
        if site_index == 'unit_cell_all':
            struct_radii = self.all_vis_radii[self.istruct]
            for isite, _site in enumerate(self.current_structure):
                vis_radius = 1.5 * tag.get('radius', struct_radii[isite])
                tags[isite, (0, 0, 0)] = {'radius': vis_radius, 'color': color, 'opacity': opacity}
            continue
        cell_index = tag['cell_index']
        if 'radius' in tag:
            vis_radius = tag['radius']
        elif 'radius_factor' in tag:
            vis_radius = tag['radius_factor'] * self.all_vis_radii[self.istruct][site_index]
        else:
            vis_radius = 1.5 * self.all_vis_radii[self.istruct][site_index]
        tags[site_index, cell_index] = {'radius': vis_radius, 'color': color, 'opacity': opacity}
    for site_and_cell_index, tag_style in tags.items():
        isite, cell_index = site_and_cell_index
        site = self.current_structure[isite]
        if cell_index == (0, 0, 0):
            coords = site.coords
        else:
            fcoords = site.frac_coords + np.array(cell_index)
            site_image = PeriodicSite(site.species, fcoords, self.current_structure.lattice, to_unit_cell=False, coords_are_cartesian=False, properties=site.properties)
            self.add_site(site_image)
            coords = site_image.coords
        vis_radius = tag_style['radius']
        color = tag_style['color']
        opacity = tag_style['opacity']
        self.add_partial_sphere(coords=coords, radius=vis_radius, color=color, start=0, end=360, opacity=opacity)