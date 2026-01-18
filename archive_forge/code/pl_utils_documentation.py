from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt

    >>> o  = Vector3([0, 0, 0])
    >>> a1 = Vector3([1, 0, 0])
    >>> a2 = Vector3([2, 0, 0])
    >>> a3 = Vector3([3, 0, 0])
    >>> a4 = Vector3([4, 0, 0])
    >>> b0 = Vector3([0, 2, 0])
    >>> b1 = Vector3([1, 2, 0])
    >>> b2 = Vector3([2, 2, 0])
    >>> b3 = Vector3([3, 2, 0])
    >>> b4 = Vector3([4, 2, 0])
    >>> c1 = Vector3([0, 0, 1])
    >>> arc_distance_sq_checked([o, a1], [o, b0])
    0
    >>> arc_distance_sq_checked([o, a1], [c1, a1 + c1])
    1
    >>> arc_b = [Vector3([1, 1, -1]), Vector3([1, 1, 1])]
    >>> arc_distance_sq_checked([-c1, c1], arc_b)
    2

    Now some cases were everything is on one line.

    >>> arc_distance_sq_checked([o, a3], [a1, a2])
    0
    >>> arc_distance_sq_checked([o, a2], [a1, a3])
    0
    >>> arc_distance_sq_checked([o, a1], [a3, a4])
    4
    >>> arc_distance_sq_checked([o, a1], [a1/2, 2*a1])
    0

    Arcs are parallel but on distinct lines

    >>> arc_distance_sq_checked([b0, b1], [a3, a4])
    8
    >>> arc_distance_sq_checked([b0, b4], [a2, a3])
    4
    >>> arc_distance_sq_checked([b0, b1], [a1, a2])
    4

    Now some more generic cases

    >>> half = 1/QQ(2)
    >>> arc_b = [Vector3([0, 1, half]), Vector3([1, 0, half])]
    >>> arc_distance_sq_checked([o, c1], arc_b) == half
    True
    >>> arc_b = [Vector3([ 1, 1, 0]), Vector3([0, 1, 0])]
    >>> arc_distance_sq_checked([-a1, o], arc_b)
    1
    >>> arc_b = [Vector3([-1, 1, 0]), Vector3([2, 1, 0])]
    >>> arc_distance_sq_checked([o, a1], arc_b)
    1
    >>> arc_b = [Vector3([-1, -1, 1]), Vector3([1, 1, 1])]
    >>> arc_distance_sq_checked([-a1, a1], arc_b)
    1
    >>> arc_b = [Vector3([1, 0, 1]), Vector3([2, -1, 2])]
    >>> arc_distance_sq_checked([o, a2], arc_b)
    1
    >>> arc_b = [Vector3([1, 0, 1]), Vector3([2, -1, 3])]
    >>> arc_distance_sq_checked([o, a2], arc_b)
    1

    