import gyp.easy_xml as easy_xml
def _AddFilesToNode(self, parent, files):
    """Adds files and/or filters to the parent node.

    Args:
      parent: Destination node
      files: A list of Filter objects and/or relative paths to files.

    Will call itself recursively, if the files list contains Filter objects.
    """
    for f in files:
        if isinstance(f, Filter):
            node = ['Filter', {'Name': f.name}]
            self._AddFilesToNode(node, f.contents)
        else:
            node = ['File', {'RelativePath': f}]
            self.files_dict[f] = node
        parent.append(node)