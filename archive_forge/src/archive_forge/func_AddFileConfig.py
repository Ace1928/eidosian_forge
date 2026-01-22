import gyp.easy_xml as easy_xml
def AddFileConfig(self, path, config, attrs=None, tools=None):
    """Adds a configuration to a file.

    Args:
      path: Relative path to the file.
      config: Name of configuration to add.
      attrs: Dict of configuration attributes; may be None.
      tools: List of tools (strings or Tool objects); may be None.

    Raises:
      ValueError: Relative path does not match any file added via AddFiles().
    """
    parent = self.files_dict.get(path)
    if not parent:
        raise ValueError('AddFileConfig: file "%s" not in project.' % path)
    spec = self._GetSpecForConfiguration('FileConfiguration', config, attrs, tools)
    parent.append(spec)