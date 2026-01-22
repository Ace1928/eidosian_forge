from typing import TYPE_CHECKING, Any, MutableMapping, Optional
def _load_page(self):
    if not self.more:
        return False
    self.update_variables()
    self.last_response = self.client.execute(self.QUERY, variable_values=self.variables)
    self.objects.extend(self.convert_objects())
    return True