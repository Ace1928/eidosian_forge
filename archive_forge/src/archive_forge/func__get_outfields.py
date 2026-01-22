from ..base import traits, TraitedSpec, DynamicTraitedSpec, File, BaseInterface
from ..io import add_traits
def _get_outfields(self):
    with open(self.inputs.in_file, 'r') as fid:
        entry = self._parse_line(fid.readline())
        if self.inputs.header:
            self._outfields = tuple(entry)
        else:
            self._outfields = tuple(['column_' + str(x) for x in range(len(entry))])
    return self._outfields