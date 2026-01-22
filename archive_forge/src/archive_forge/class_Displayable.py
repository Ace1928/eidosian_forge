import json
import pkgutil
import textwrap
from typing import Callable, Dict, Optional, Tuple, Any, Union
import uuid
from ._vegafusion_data import compile_with_vegafusion, using_vegafusion
from .plugin_registry import PluginRegistry, PluginEnabler
from .mimebundle import spec_to_mimebundle
from .schemapi import validate_jsonschema
class Displayable:
    """A base display class for VegaLite v1/v2.

    This class takes a VegaLite v1/v2 spec and does the following:

    1. Optionally validates the spec against a schema.
    2. Uses the RendererPlugin to grab a renderer and call it when the
       IPython/Jupyter display method (_repr_mimebundle_) is called.

    The spec passed to this class must be fully schema compliant and already
    have the data portion of the spec fully processed and ready to serialize.
    In practice, this means, the data portion of the spec should have been passed
    through appropriate data model transformers.
    """
    renderers: Optional[RendererRegistry] = None
    schema_path = ('altair', '')

    def __init__(self, spec: dict, validate: bool=False) -> None:
        self.spec = spec
        self.validate = validate
        self._validate()

    def _validate(self) -> None:
        """Validate the spec against the schema."""
        data = pkgutil.get_data(*self.schema_path)
        assert data is not None
        schema_dict: dict = json.loads(data.decode('utf-8'))
        validate_jsonschema(self.spec, schema_dict)

    def _repr_mimebundle_(self, include: Any=None, exclude: Any=None) -> MimeBundleType:
        """Return a MIME bundle for display in Jupyter frontends."""
        if self.renderers is not None:
            renderer_func = self.renderers.get()
            assert renderer_func is not None
            return renderer_func(self.spec)
        else:
            return {}