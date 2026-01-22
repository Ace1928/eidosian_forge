from .atom  import atom_add_entry_type
from .html5 import html5_extra_attributes, remove_rel
class HostLanguage:
    """An enumeration style class: recognized host language types for this processor of RDFa. Some processing details may depend on these host languages. "rdfa_core" is the default Host Language is nothing else is defined."""
    rdfa_core = 'RDFa Core'
    xhtml = 'XHTML+RDFa'
    xhtml5 = 'XHTML5+RDFa'
    html5 = 'HTML5+RDFa'
    atom = 'Atom+RDFa'
    svg = 'SVG+RDFa'