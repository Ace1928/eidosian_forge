from typing import List, TypeVar

    Wrapper class around the base Python `List` type. Contains an additional `token`  string
    attribute that can be passed to the pagination API that returned this list to fetch additional
    elements, if any are available
    