import pathlib
import param
from ...theme import Design
from ...theme.material import Material
from ..base import BasicTemplate, TemplateActions
class MaterialTemplate(BasicTemplate):
    """
    MaterialTemplate is built on top of Material web components.
    """
    design = param.ClassSelector(class_=Design, default=Material, is_instance=False, instantiate=False, doc='\n        A Design applies a specific design system to a template.')
    sidebar_width = param.Integer(default=370, doc='\n        The width of the sidebar in pixels. Default is 370.')
    _actions = param.ClassSelector(default=MaterialTemplateActions(), class_=TemplateActions)
    _css = [_ROOT / 'material.css']
    _template = _ROOT / 'material.html'