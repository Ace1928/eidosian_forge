from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
class SphericalJoint(Joint):
    """Spherical (Ball-and-Socket) Joint.

    .. image:: SphericalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A spherical joint is defined such that the child body is free to rotate in
    any direction, without allowing a translation of the ``child_point``. As can
    also be seen in the image, the ``parent_point`` and ``child_point`` are
    fixed on top of each other, i.e. the ``joint_point``. This rotation is
    defined using the :func:`parent_interframe.orient(child_interframe,
    rot_type, amounts, rot_order)
    <sympy.physics.vector.frame.ReferenceFrame.orient>` method. The default
    rotation consists of three relative rotations, i.e. body-fixed rotations.
    Based on the direction cosine matrix following from these rotations, the
    angular velocity is computed based on the generalized coordinates and
    generalized speeds.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Body
        The parent body of joint.
    child : Body
        The child body of joint.
    coordinates: iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    rot_type : str, optional
        The method used to generate the direction cosine matrix. Supported
        methods are:

        - ``'Body'``: three successive rotations about new intermediate axes,
          also called "Euler and Tait-Bryan angles"
        - ``'Space'``: three successive rotations about the parent frames' unit
          vectors

        The default method is ``'Body'``.
    amounts :
        Expressions defining the rotation angles or direction cosine matrix.
        These must match the ``rot_type``. See examples below for details. The
        input types are:

        - ``'Body'``: 3-tuple of expressions, symbols, or functions
        - ``'Space'``: 3-tuple of expressions, symbols, or functions

        The default amounts are the given ``coordinates``.
    rot_order : str or int, optional
        If applicable, the order of the successive of rotations. The string
        ``'123'`` and integer ``123`` are equivalent, for example. Required for
        ``'Body'`` and ``'Space'``. The default value is ``123``.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Body
        The joint's parent body.
    child : Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single spherical joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import Body, SphericalJoint
    >>> parent = Body('P')
    >>> parent
    P
    >>> child = Body('C')
    >>> child
    C
    >>> joint = SphericalJoint('PC', parent, child)
    >>> joint
    SphericalJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_interframe
    P_frame
    >>> joint.child_interframe
    C_frame
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame).to_matrix(child.frame)
    Matrix([
    [ u0_PC(t)*cos(q1_PC(t))*cos(q2_PC(t)) + u1_PC(t)*sin(q2_PC(t))],
    [-u0_PC(t)*sin(q2_PC(t))*cos(q1_PC(t)) + u1_PC(t)*cos(q2_PC(t))],
    [                             u0_PC(t)*sin(q1_PC(t)) + u2_PC(t)]])
    >>> child.frame.x.to_matrix(parent.frame)
    Matrix([
    [                                            cos(q1_PC(t))*cos(q2_PC(t))],
    [sin(q0_PC(t))*sin(q1_PC(t))*cos(q2_PC(t)) + sin(q2_PC(t))*cos(q0_PC(t))],
    [sin(q0_PC(t))*sin(q2_PC(t)) - sin(q1_PC(t))*cos(q0_PC(t))*cos(q2_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the spherical joint, the kinematics of a
    spherical joint with a ZXZ rotation can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Body, SphericalJoint
    >>> l1 = symbols('l1')

    First create bodies to represent the fixed floor and a pendulum bob.

    >>> floor = Body('F')
    >>> bob = Body('B')

    The joint will connect the bob to the floor, with the joint located at a
    distance of ``l1`` from the child's center of mass and the rotation set to a
    body-fixed ZXZ rotation.

    >>> joint = SphericalJoint('S', floor, bob, child_point=l1 * bob.y,
    ...                        rot_type='body', rot_order='ZXZ')

    Now that the joint is established, the kinematics of the connected body can
    be accessed.

    The position of the bob's masscenter is found with:

    >>> bob.masscenter.pos_from(floor.masscenter)
    - l1*B_frame.y

    The angular velocities of the pendulum link can be computed with respect to
    the floor.

    >>> bob.frame.ang_vel_in(floor.frame).to_matrix(
    ...     floor.frame).simplify()
    Matrix([
    [u1_S(t)*cos(q0_S(t)) + u2_S(t)*sin(q0_S(t))*sin(q1_S(t))],
    [u1_S(t)*sin(q0_S(t)) - u2_S(t)*sin(q1_S(t))*cos(q0_S(t))],
    [                          u0_S(t) + u2_S(t)*cos(q1_S(t))]])

    Finally, the linear velocity of the bob's center of mass can be computed.

    >>> bob.masscenter.vel(floor.frame).to_matrix(bob.frame)
    Matrix([
    [                           l1*(u0_S(t)*cos(q1_S(t)) + u2_S(t))],
    [                                                             0],
    [-l1*(u0_S(t)*sin(q1_S(t))*sin(q2_S(t)) + u1_S(t)*cos(q2_S(t)))]])

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None, parent_point=None, child_point=None, parent_interframe=None, child_interframe=None, rot_type='BODY', amounts=None, rot_order=123):
        self._rot_type = rot_type
        self._amounts = amounts
        self._rot_order = rot_order
        super().__init__(name, parent, child, coordinates, speeds, parent_point, child_point, parent_interframe=parent_interframe, child_interframe=child_interframe)

    def __str__(self):
        return f'SphericalJoint: {self.name}  parent: {self.parent}  child: {self.child}'

    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 3, 'q')

    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, len(self.coordinates), 'u')

    def _orient_frames(self):
        supported_rot_types = ('BODY', 'SPACE')
        if self._rot_type.upper() not in supported_rot_types:
            raise NotImplementedError(f'Rotation type "{self._rot_type}" is not implemented. Implemented rotation types are: {supported_rot_types}')
        amounts = self.coordinates if self._amounts is None else self._amounts
        self.child_interframe.orient(self.parent_interframe, self._rot_type, amounts, self._rot_order)

    def _set_angular_velocity(self):
        t = dynamicsymbols._t
        vel = self.child_interframe.ang_vel_in(self.parent_interframe).xreplace({q.diff(t): u for q, u in zip(self.coordinates, self.speeds)})
        self.child_interframe.set_ang_vel(self.parent_interframe, vel)

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self.parent.frame, 0)
        self.child_point.set_vel(self.child.frame, 0)
        self.child.masscenter.v2pt_theory(self.parent_point, self.parent.frame, self.child.frame)