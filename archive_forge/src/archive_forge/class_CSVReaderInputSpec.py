from ..base import traits, TraitedSpec, DynamicTraitedSpec, File, BaseInterface
from ..io import add_traits
class CSVReaderInputSpec(DynamicTraitedSpec, TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc='Input comma-seperated value (CSV) file')
    header = traits.Bool(False, usedefault=True, desc='True if the first line is a column header')