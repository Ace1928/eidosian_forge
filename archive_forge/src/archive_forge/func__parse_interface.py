import re
from sphinx.locale import _
from sphinx.ext.napoleon.docstring import NumpyDocstring
def _parse_interface(obj):
    """Print description for input parameters."""
    parsed = []
    if obj.input_spec:
        inputs = obj.input_spec()
        mandatory_items = sorted(inputs.traits(mandatory=True).items())
        if mandatory_items:
            parsed += ['', 'Mandatory Inputs']
            parsed += ['-' * len(parsed[-1])]
            for name, spec in mandatory_items:
                parsed += _parse_spec(inputs, name, spec)
        mandatory_keys = {item[0] for item in mandatory_items}
        optional_items = sorted([(name, val) for name, val in inputs.traits(transient=None).items() if name not in mandatory_keys])
        if optional_items:
            parsed += ['', 'Optional Inputs']
            parsed += ['-' * len(parsed[-1])]
            for name, spec in optional_items:
                parsed += _parse_spec(inputs, name, spec)
    if obj.output_spec:
        outputs = sorted(obj.output_spec().traits(transient=None).items())
        if outputs:
            parsed += ['', 'Outputs']
            parsed += ['-' * len(parsed[-1])]
            for name, spec in outputs:
                parsed += _parse_spec(inputs, name, spec)
    return parsed