from .vector import Vector, _check_vector
from .frame import _check_frame
from warnings import warn
def vel(self, frame):
    """The velocity Vector of this Point in the ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which the returned velocity vector will be defined in

        Examples
        ========

        >>> from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
        >>> N = ReferenceFrame('N')
        >>> p1 = Point('p1')
        >>> p1.set_vel(N, 10 * N.x)
        >>> p1.vel(N)
        10*N.x

        Velocities will be automatically calculated if possible, otherwise a
        ``ValueError`` will be returned. If it is possible to calculate
        multiple different velocities from the relative points, the points
        defined most directly relative to this point will be used. In the case
        of inconsistent relative positions of points, incorrect velocities may
        be returned. It is up to the user to define prior relative positions
        and velocities of points in a self-consistent way.

        >>> p = Point('p')
        >>> q = dynamicsymbols('q')
        >>> p.set_vel(N, 10 * N.x)
        >>> p2 = Point('p2')
        >>> p2.set_pos(p, q*N.x)
        >>> p2.vel(N)
        (Derivative(q(t), t) + 10)*N.x

        """
    _check_frame(frame)
    if not frame in self._vel_dict:
        valid_neighbor_found = False
        is_cyclic = False
        visited = []
        queue = [self]
        candidate_neighbor = []
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                for neighbor, neighbor_pos in node._pos_dict.items():
                    if neighbor in visited:
                        continue
                    try:
                        neighbor_pos.express(frame)
                    except ValueError:
                        continue
                    if neighbor in queue:
                        is_cyclic = True
                    try:
                        neighbor_velocity = neighbor._vel_dict[frame]
                    except KeyError:
                        queue.append(neighbor)
                        continue
                    candidate_neighbor.append(neighbor)
                    if not valid_neighbor_found:
                        self.set_vel(frame, self.pos_from(neighbor).dt(frame) + neighbor_velocity)
                        valid_neighbor_found = True
        if is_cyclic:
            warn('Kinematic loops are defined among the positions of points. This is likely not desired and may cause errors in your calculations.')
        if len(candidate_neighbor) > 1:
            warn('Velocity automatically calculated based on point ' + candidate_neighbor[0].name + ' but it is also possible from points(s):' + str(candidate_neighbor[1:]) + '. Velocities from these points are not necessarily the same. This may cause errors in your calculations.')
        if valid_neighbor_found:
            return self._vel_dict[frame]
        else:
            raise ValueError('Velocity of point ' + self.name + ' has not been defined in ReferenceFrame ' + frame.name)
    return self._vel_dict[frame]