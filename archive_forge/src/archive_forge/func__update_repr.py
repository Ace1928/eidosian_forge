from ase import Atoms
def _update_repr(self, chg=None):
    self.view.update_spacefill(radiusType='covalent', radiusScale=self.rad.value, color_scheme=self.csel.value, color_scale='rainbow')