def toHexRGB(self):
    """Convert the color back to an integer suitable for the """
    '0xRRGGBB hex representation'
    r = int(255 * self.red)
    g = int(255 * self.green)
    b = int(255 * self.blue)
    return (r << 16) + (g << 8) + b