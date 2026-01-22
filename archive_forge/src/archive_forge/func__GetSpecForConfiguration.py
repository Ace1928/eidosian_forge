import gyp.easy_xml as easy_xml
def _GetSpecForConfiguration(self, config_type, config_name, attrs, tools):
    """Returns the specification for a configuration.

    Args:
      config_type: Type of configuration node.
      config_name: Configuration name.
      attrs: Dict of configuration attributes; may be None.
      tools: List of tools (strings or Tool objects); may be None.
    Returns:
    """
    if not attrs:
        attrs = {}
    if not tools:
        tools = []
    node_attrs = attrs.copy()
    node_attrs['Name'] = config_name
    specification = [config_type, node_attrs]
    if tools:
        for t in tools:
            if isinstance(t, Tool):
                specification.append(t._GetSpecification())
            else:
                specification.append(Tool(t)._GetSpecification())
    return specification