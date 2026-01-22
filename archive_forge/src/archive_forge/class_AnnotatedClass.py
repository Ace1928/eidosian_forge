class AnnotatedClass:
    """
    A class with annotated methods.
    """

    def __init__(self, v: int):
        self.x = v

    def add(self, v: int) -> int:
        return self.x + v