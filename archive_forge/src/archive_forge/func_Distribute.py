from sys import version_info as _swig_python_version_info
import weakref
def Distribute(self, *args):
    """
        *Overload 1:*
        Aggregated version of count:  |{i | v[i] == values[j]}| == cards[j]

        |

        *Overload 2:*
        Aggregated version of count:  |{i | v[i] == values[j]}| == cards[j]

        |

        *Overload 3:*
        Aggregated version of count:  |{i | v[i] == j}| == cards[j]

        |

        *Overload 4:*
        Aggregated version of count with bounded cardinalities:
        forall j in 0 .. card_size - 1: card_min <= |{i | v[i] == j}| <= card_max

        |

        *Overload 5:*
        Aggregated version of count with bounded cardinalities:
        forall j in 0 .. card_size - 1:
           card_min[j] <= |{i | v[i] == j}| <= card_max[j]

        |

        *Overload 6:*
        Aggregated version of count with bounded cardinalities:
        forall j in 0 .. card_size - 1:
           card_min[j] <= |{i | v[i] == j}| <= card_max[j]

        |

        *Overload 7:*
        Aggregated version of count with bounded cardinalities:
        forall j in 0 .. card_size - 1:
           card_min[j] <= |{i | v[i] == values[j]}| <= card_max[j]

        |

        *Overload 8:*
        Aggregated version of count with bounded cardinalities:
        forall j in 0 .. card_size - 1:
           card_min[j] <= |{i | v[i] == values[j]}| <= card_max[j]
        """
    return _pywrapcp.Solver_Distribute(self, *args)