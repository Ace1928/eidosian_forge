import inspect
import textwrap
import param
import panel as _pn
import holoviews as _hv
from holoviews import Store, render  # noqa
from .converter import HoloViewsConverter
from .interactive import Interactive
from .ui import explorer  # noqa
from .utilities import hvplot_extension, output, save, show # noqa
from .plotting import (hvPlot, hvPlotTabular,  # noqa
def _get_doc_and_signature(cls, kind, completions=False, docstring=True, generic=True, style=True, signature=None):
    converter = HoloViewsConverter
    method = getattr(cls, kind)
    kind_opts = converter._kind_options.get(kind, [])
    eltype = converter._kind_mapping[kind]
    formatter = ''
    if completions:
        formatter = 'hvplot.{kind}({completions})'
    if docstring:
        if formatter:
            formatter += '\n'
        formatter += '{docstring}'
    if generic:
        if formatter:
            formatter += '\n'
        formatter += '{options}'
    backend = hvplot_extension.compatibility or Store.current_backend
    if eltype in Store.registry[backend]:
        valid_opts = Store.registry[backend][eltype].style_opts
        if style:
            formatter += '\n{style}'
    else:
        valid_opts = []
    style_opts = 'Style options\n-------------\n\n' + '\n'.join(sorted(valid_opts))
    parameters = []
    extra_kwargs = _hv.core.util.unique_iterator(valid_opts + kind_opts + converter._axis_options + converter._op_options)
    sig = signature or inspect.signature(method)
    for name, p in list(sig.parameters.items())[1:]:
        if p.kind == 1:
            parameters.append((name, p.default))
    filtered_signature = [p for p in sig.parameters.values() if p.kind != inspect.Parameter.VAR_KEYWORD]
    extra_params = [inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY) for k in extra_kwargs if k not in [p.name for p in filtered_signature]]
    all_params = filtered_signature + extra_params + [inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)]
    signature = inspect.Signature(all_params)
    parameters += [(o, None) for o in extra_kwargs]
    completions = ', '.join([f'{n}={v}' for n, v in parameters])
    options = textwrap.dedent(converter.__doc__)
    method_doc = _METHOD_DOCS.get(kind, method.__doc__)
    _METHOD_DOCS[kind] = method_doc
    docstring = formatter.format(kind=kind, completions=completions, docstring=textwrap.dedent(method_doc), options=options, style=style_opts)
    return (docstring, signature)