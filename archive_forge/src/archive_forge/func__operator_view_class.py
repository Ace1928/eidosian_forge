import importlib
import inspect
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