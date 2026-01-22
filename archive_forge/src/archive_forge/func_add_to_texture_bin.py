def add_to_texture_bin(self, texture_bin, border=0):
    """Add the images of the animation to a :py:class:`~pyglet.image.atlas.TextureBin`.

        The animation frames are modified in-place to refer to the texture bin
        regions.

        :Parameters:
            `texture_bin` : `~pyglet.image.atlas.TextureBin`
                Texture bin to upload animation frames into.
            `border` : int
                Leaves specified pixels of blank space around
                each image frame when adding to the TextureBin.

        """
    for frame in self.frames:
        frame.image = texture_bin.add(frame.image, border)