from __future__ import annotations
class ChemenvError(Exception):
    """Chemenv error."""

    def __init__(self, cls: str, method: str, msg: str):
        """
        Args:
            cls:
            method:
            msg:
        """
        self.cls = cls
        self.method = method
        self.msg = msg

    def __str__(self):
        return f'{self.cls}: {self.method}\n{self.msg}'