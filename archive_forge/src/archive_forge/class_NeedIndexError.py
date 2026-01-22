class NeedIndexError(Error):
    """No matching index was found for a query that requires an index. Check
  the Indexes page in the Admin Console and your index.yaml file.
  """

    def __init__(self, error, original_message=None, header=None, yaml_index=None, xml_index=None):
        super(NeedIndexError, self).__init__(error)
        self._original_message = original_message
        self._header = header
        self._yaml_index = yaml_index
        self._xml_index = xml_index

    def OriginalMessage(self):
        return self._original_message

    def Header(self):
        return self._header

    def YamlIndex(self):
        return self._yaml_index

    def XmlIndex(self):
        return self._xml_index