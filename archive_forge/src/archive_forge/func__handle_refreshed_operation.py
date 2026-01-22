import threading
from google.api_core import exceptions
from google.api_core.future import polling
def _handle_refreshed_operation(self):
    with self._completion_lock:
        if not self._extended_operation.done:
            return
        if self.error_code and self.error_message:
            errors = []
            if hasattr(self, 'error') and hasattr(self.error, 'errors'):
                errors = self.error.errors
            exception = exceptions.from_http_status(status_code=self.error_code, message=self.error_message, response=self._extended_operation, errors=errors)
            self.set_exception(exception)
        elif self.error_code or self.error_message:
            exception = exceptions.GoogleAPICallError(f'Unexpected error {self.error_code}: {self.error_message}')
            self.set_exception(exception)
        else:
            self.set_result(None)