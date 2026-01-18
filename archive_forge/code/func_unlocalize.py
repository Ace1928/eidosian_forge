from django.template import Library, Node, TemplateSyntaxError
from django.utils import formats
@register.filter(is_safe=False)
def unlocalize(value):
    """
    Force a value to be rendered as a non-localized value.
    """
    return str(formats.localize(value, use_l10n=False))