import copy
from ..core.pattern import Pattern
def __set_templated_pattern(self):
    items = list()
    if not self._disabled.php:
        items.append(self.__patterns.php._starting_pattern.pattern)
    if not self._disabled.handlebars:
        items.append(self.__patterns.handlebars._starting_pattern.pattern)
    if not self._disabled.erb:
        items.append(self.__patterns.erb._starting_pattern.pattern)
    if not self._disabled.django:
        items.append(self.__patterns.django._starting_pattern.pattern)
        items.append(self.__patterns.django_value._starting_pattern.pattern)
        items.append(self.__patterns.django_comment._starting_pattern.pattern)
    if not self._disabled.smarty:
        items.append(self.__patterns.smarty._starting_pattern.pattern)
    if self._until_pattern:
        items.append(self._until_pattern.pattern)
    self.__template_pattern = self._input.get_regexp('(?:' + '|'.join(items) + ')')