import textwrap
import pkgutil
import copy
import os
import json
from functools import reduce
class TemplatesConfig(object):
    """
    Singleton object containing the current figure templates (aka themes)
    """

    def __init__(self):
        self._templates = {}
        default_templates = ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
        for template_name in default_templates:
            self._templates[template_name] = Lazy
        self._validator = None
        self._default = None

    def __len__(self):
        return len(self._templates)

    def __contains__(self, item):
        return item in self._templates

    def __iter__(self):
        return iter(self._templates)

    def __getitem__(self, item):
        if isinstance(item, str):
            template_names = item.split('+')
        else:
            template_names = [item]
        templates = []
        for template_name in template_names:
            template = self._templates[template_name]
            if template is Lazy:
                from plotly.graph_objs.layout import Template
                if template_name == 'none':
                    template = Template(data_scatter=[{}])
                    self._templates[template_name] = template
                else:
                    path = os.path.join('package_data', 'templates', template_name + '.json')
                    template_str = pkgutil.get_data('plotly', path).decode('utf-8')
                    template_dict = json.loads(template_str)
                    template = Template(template_dict, _validate=False)
                    self._templates[template_name] = template
            templates.append(self._templates[template_name])
        return self.merge_templates(*templates)

    def __setitem__(self, key, value):
        self._templates[key] = self._validate(value)

    def __delitem__(self, key):
        del self._templates[key]
        if self._default == key:
            self._default = None

    def _validate(self, value):
        if not self._validator:
            from plotly.validators.layout import TemplateValidator
            self._validator = TemplateValidator()
        return self._validator.validate_coerce(value)

    def keys(self):
        return self._templates.keys()

    def items(self):
        return self._templates.items()

    def update(self, d={}, **kwargs):
        """
        Update one or more templates from a dict or from input keyword
        arguments.

        Parameters
        ----------
        d: dict
            Dictionary from template names to new template values.

        kwargs
            Named argument value pairs where the name is a template name
            and the value is a new template value.
        """
        for k, v in dict(d, **kwargs).items():
            self[k] = v

    @property
    def default(self):
        """
        The name of the default template, or None if no there is no default

        If not None, the default template is automatically applied to all
        figures during figure construction if no explicit template is
        specified.

        The names of available templates may be retrieved with:

        >>> import plotly.io as pio
        >>> list(pio.templates)

        Returns
        -------
        str
        """
        return self._default

    @default.setter
    def default(self, value):
        self._validate(value)
        self._default = value

    def __repr__(self):
        return 'Templates configuration\n-----------------------\n    Default template: {default}\n    Available templates:\n{available}\n'.format(default=repr(self.default), available=self._available_templates_str())

    def _available_templates_str(self):
        """
        Return nicely wrapped string representation of all
        available template names
        """
        available = '\n'.join(textwrap.wrap(repr(list(self)), width=79 - 8, initial_indent=' ' * 8, subsequent_indent=' ' * 9))
        return available

    def merge_templates(self, *args):
        """
        Merge a collection of templates into a single combined template.
        Templates are process from left to right so if multiple templates
        specify the same propery, the right-most template will take
        precedence.

        Parameters
        ----------
        args: list of Template
            Zero or more template objects (or dicts with compatible properties)

        Returns
        -------
        template:
            A combined template object

        Examples
        --------

        >>> pio.templates.merge_templates(
        ...     go.layout.Template(layout={'font': {'size': 20}}),
        ...     go.layout.Template(data={'scatter': [{'mode': 'markers'}]}),
        ...     go.layout.Template(layout={'font': {'family': 'Courier'}}))
        layout.Template({
            'data': {'scatter': [{'mode': 'markers', 'type': 'scatter'}]},
            'layout': {'font': {'family': 'Courier', 'size': 20}}
        })
        """
        if args:
            return reduce(self._merge_2_templates, args)
        else:
            from plotly.graph_objs.layout import Template
            return Template()

    def _merge_2_templates(self, template1, template2):
        """
        Helper function for merge_templates that merges exactly two templates

        Parameters
        ----------
        template1: Template
        template2: Template

        Returns
        -------
        Template:
            merged template
        """
        result = self._validate(template1)
        other = self._validate(template2)
        for trace_type in result.data:
            result_traces = result.data[trace_type]
            other_traces = other.data[trace_type]
            if result_traces and other_traces:
                lcm = len(result_traces) * len(other_traces) // gcd(len(result_traces), len(other_traces))
                result.data[trace_type] = result_traces * (lcm // len(result_traces))
                other.data[trace_type] = other_traces * (lcm // len(other_traces))
        result.update(other)
        return result