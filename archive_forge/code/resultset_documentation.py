from boto.s3.user import User

    The ResultSet is used to pass results back from the Amazon services
    to the client. It is light wrapper around Python's :py:class:`list` class,
    with some additional methods for parsing XML results from AWS.
    Because I don't really want any dependencies on external libraries,
    I'm using the standard SAX parser that comes with Python. The good news is
    that it's quite fast and efficient but it makes some things rather
    difficult.

    You can pass in, as the marker_elem parameter, a list of tuples.
    Each tuple contains a string as the first element which represents
    the XML element that the resultset needs to be on the lookout for
    and a Python class as the second element of the tuple. Each time the
    specified element is found in the XML, a new instance of the class
    will be created and popped onto the stack.

    :ivar str next_token: A hash used to assist in paging through very long
        result sets. In most cases, passing this value to certain methods
        will give you another 'page' of results.
    