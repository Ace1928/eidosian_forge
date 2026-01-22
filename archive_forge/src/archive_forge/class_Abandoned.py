class Abandoned(Exception):
    """Indicates that some computation is being abandoned.

    Abandoning a computation is different than returning a value or raising
    an exception indicating some operational or programming defect.
    """