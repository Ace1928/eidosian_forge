from OpenGL._bytes import integer_types
def uintToLong(value):
    if value < 0:
        value = (value & 2147483647) + 2147483648
    return value