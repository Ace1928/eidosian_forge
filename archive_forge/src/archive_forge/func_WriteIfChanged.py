import gyp.easy_xml as easy_xml
def WriteIfChanged(self):
    """Writes the tool file."""
    content = ['VisualStudioToolFile', {'Version': '8.00', 'Name': self.name}, self.rules_section]
    easy_xml.WriteXmlIfChanged(content, self.tool_file_path, encoding='Windows-1252')