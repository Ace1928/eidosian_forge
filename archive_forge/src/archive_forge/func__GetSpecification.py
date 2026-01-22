import gyp.easy_xml as easy_xml
def _GetSpecification(self):
    """Creates an element for the tool.

    Returns:
      A new xml.dom.Element for the tool.
    """
    return ['Tool', self._attrs]