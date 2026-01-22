from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def arc_distance_sq(arc_a, arc_b):
    """
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

    """
    a0, a1 = arc_a
    b0, b1 = arc_b
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    if are_parallel(u, v) and are_parallel(u, w):
        U = Matrix([u]).transpose()
        t0, t1 = (U.solve_right(b0 - a0)[0], U.solve_right(b1 - a0)[0])
        if t0 > t1:
            t0, t1 = (t1, t0)
        if t1 < 0:
            return t1 ** 2 * norm_sq(u)
        elif 1 < t0:
            return (t0 - 1) ** 2 * norm_sq(u)
        return 0
    X = Matrix([[u * u, -u * v], [u * v, -v * v]])
    Y = Vector2([-u * w, -v * w])
    if X.det() == 0:
        t, s = (-(w * u) / norm_sq(u), 0)
    else:
        t, s = X.solve_right(Y)
    if 0 <= t <= 1 and 0 <= s <= 1:
        pA = a0 + Vector3(t * u)
        pB = b0 + Vector3(s * v)
        return norm_sq(pA - pB)
    t = (b0 - a0) * u / norm_sq(u)
    s = (b1 - a0) * u / norm_sq(u)
    x = (a0 - b0) * v / norm_sq(v)
    y = (a1 - b0) * v / norm_sq(v)
    s = min(max(s, 0), 1)
    t = min(max(t, 0), 1)
    x = min(max(x, 0), 1)
    y = min(max(y, 0), 1)
    p = a0 + Vector3(t * u)
    q = a0 + Vector3(s * u)
    f = b0 + Vector3(x * v)
    g = b0 + Vector3(y * v)
    return min([norm_sq(p - b0), norm_sq(q - b1), norm_sq(f - a0), norm_sq(g - a1)])