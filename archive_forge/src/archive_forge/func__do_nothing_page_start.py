import abc
def _do_nothing_page_start(iterator, page, response):
    """Helper to provide custom behavior after a :class:`Page` is started.

    This is a do-nothing stand-in as the default value.

    Args:
        iterator (Iterator): An iterator that holds some request info.
        page (Page): The page that was just created.
        response (Any): The API response for a page.
    """
    pass