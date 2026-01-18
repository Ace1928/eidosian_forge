from boto import handler
import xml.sax

    The status returned from a MultiObject Delete request.

    :ivar deleted: A list of successfully deleted objects.  Note that if
        the quiet flag was specified in the request, this list will
        be empty because only error responses would be returned.

    :ivar errors: A list of unsuccessfully deleted objects.
    