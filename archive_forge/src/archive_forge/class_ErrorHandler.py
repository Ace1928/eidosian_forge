import xml.sax
import xml.sax.handler
class ErrorHandler:

    def warning(self, exception):
        print(exception)

    def error(self, exception):
        raise exception

    def fatalError(self, exception):
        raise exception