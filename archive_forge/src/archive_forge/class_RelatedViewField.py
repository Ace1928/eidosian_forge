import importlib
import inspect
class RelatedViewField(Field):

    def __init__(self, field_name, operator_view_class):
        super(RelatedViewField, self).__init__(field_name)
        self.__operator_view_class = operator_view_class

    @property
    def _operator_view_class(self):
        if inspect.isclass(self.__operator_view_class):
            return self.__operator_view_class
        elif isinstance(self.__operator_view_class, str):
            try:
                module_name, class_name = self.__operator_view_class.rsplit('.', 1)
                return class_for_name(module_name, class_name)
            except (AttributeError, ValueError, ImportError):
                raise WrongOperatorViewClassError('There is no "%s" class' % self.__operator_view_class)

    def retrieve_and_wrap(self, obj):
        related_obj = self.get(obj)
        return self.wrap(related_obj)

    def wrap(self, obj):
        return self._operator_view_class(obj)