class FileExtension:
    """File Extension Metadata

    This class holds information about a image file format associated with a
    given extension. This information is used to track plugins that are known to
    be able to handle a particular format. It also contains additional
    information about a format, which is used when creating the supported format
    docs.

    Plugins known to be able to handle this format are ordered by a ``priority``
    list. This list is used to determine the ideal plugin to use when choosing a
    plugin based on file extension.

    Parameters
    ----------
    extension : str
        The name of the extension including the initial dot, e.g. ".png".
    priority : List
        A list of plugin names (entries in config.known_plugins) that can handle
        this format. The position of a plugin expresses a preference, e.g.
        ["plugin1", "plugin2"] indicates that, if available, plugin1 should be
        preferred over plugin2 when handling a request related to this format.
    name : str
        The full name of the format.
    description : str
        A description of the format.
    external_link : str
        A link to further information about the format. Typically, the format's
        specification.
    volume_support : str
        If True, the format/extension supports volumetric image data.

    Examples
    --------
    >>> FileExtension(
            name="Bitmap",
            extension=".bmp",
            priority=["pillow", "BMP-PIL", "BMP-FI", "ITK"],
            external_link="https://en.wikipedia.org/wiki/BMP_file_format",
        )

    """

    def __init__(self, *, extension, priority, name=None, description=None, external_link=None, volume_support=False):
        self.extension = extension
        self.priority = priority
        self.name = name
        self.description = description
        self.external_link = external_link
        self.default_priority = priority.copy()
        self.volume_support = volume_support

    def reset(self):
        self.priority = self.default_priority.copy()