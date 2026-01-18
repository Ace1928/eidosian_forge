from botocore.utils import is_json_value_header
def traverse_and_document_shape(self, section, shape, history, include=None, exclude=None, name=None, is_required=False):
    """Traverses and documents a shape

        Will take a self class and call its appropriate methods as a shape
        is traversed.

        :param section: The section to document.

        :param history: A list of the names of the shapes that have been
            traversed.

        :type include: Dictionary where keys are parameter names and
            values are the shapes of the parameter names.
        :param include: The parameter shapes to include in the documentation.

        :type exclude: List of the names of the parameters to exclude.
        :param exclude: The names of the parameters to exclude from
            documentation.

        :param name: The name of the shape.

        :param is_required: If the shape is a required member.
        """
    param_type = shape.type_name
    if getattr(shape, 'serialization', {}).get('eventstream'):
        param_type = 'event_stream'
    if shape.name in history:
        self.document_recursive_shape(section, shape, name=name)
    else:
        history.append(shape.name)
        is_top_level_param = len(history) == 2
        if hasattr(shape, 'is_document_type') and shape.is_document_type:
            param_type = 'document'
        getattr(self, f'document_shape_type_{param_type}', self.document_shape_default)(section, shape, history=history, name=name, include=include, exclude=exclude, is_top_level_param=is_top_level_param, is_required=is_required)
        if is_top_level_param:
            self._event_emitter.emit(f'docs.{self.EVENT_NAME}.{self._service_name}.{self._operation_name}.{name}', section=section)
        at_overlying_method_section = len(history) == 1
        if at_overlying_method_section:
            self._event_emitter.emit(f'docs.{self.EVENT_NAME}.{self._service_name}.{self._operation_name}.complete-section', section=section)
        history.pop()