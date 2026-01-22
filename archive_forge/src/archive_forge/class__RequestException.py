import json
class _RequestException(Exception):
    """Exception returned by a request."""

    def __init__(self, status, content):
        super(_RequestException, self).__init__()
        self.status = status
        self.content = content
        self.message = content
        try:
            self.message = json.loads(content)['error']['message']
        except ValueError:
            pass
        except KeyError:
            pass
        except TypeError:
            pass

    def __str__(self):
        return self.message

    @property
    def error_code(self):
        """Returns the error code if one is present and None otherwise."""
        try:
            parsed_content = json.loads(self.content)
        except ValueError:
            return None
        return parsed_content.get('error', {}).get('code', None)