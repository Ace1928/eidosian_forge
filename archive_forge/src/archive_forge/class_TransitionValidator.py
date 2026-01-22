import _plotly_utils.basevalidators
class TransitionValidator(_plotly_utils.basevalidators.CompoundValidator):

    def __init__(self, plotly_name='transition', parent_name='layout.slider', **kwargs):
        super(TransitionValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=kwargs.pop('data_class_str', 'Transition'), data_docs=kwargs.pop('data_docs', '\n            duration\n                Sets the duration of the slider transition\n            easing\n                Sets the easing function of the slider\n                transition\n'), **kwargs)