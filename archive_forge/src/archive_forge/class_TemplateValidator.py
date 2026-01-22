import _plotly_utils.basevalidators
class TemplateValidator(_plotly_utils.basevalidators.BaseTemplateValidator):

    def __init__(self, plotly_name='template', parent_name='layout', **kwargs):
        super(TemplateValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=kwargs.pop('data_class_str', 'Template'), data_docs=kwargs.pop('data_docs', '\n            data\n                :class:`plotly.graph_objects.layout.template.Da\n                ta` instance or dict with compatible properties\n            layout\n                :class:`plotly.graph_objects.Layout` instance\n                or dict with compatible properties\n'), **kwargs)