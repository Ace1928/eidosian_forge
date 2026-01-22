from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
class CylindricalJoint(Joint):
    """Cylindrical Joint.

    .. image:: CylindricalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A cylindrical joint is defined such that the child body both rotates about
    and translates along the body-fixed joint axis with respect to the parent
    body. The joint axis is both the rotation axis and translation axis. The
    location of the joint is defined by two points, one in each body, which
    coincide when the generalized coordinate corresponding to the translation is
    zero. The direction cosine matrix between the child interframe and parent
    interframe is formed using a simple rotation about the joint axis. The page
    on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Body
        The parent body of joint.
    child : Body
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    translation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the translation distance. The
        default value is ``dynamicsymbols(f'q1_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    translation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the translation velocity. The default
        value is ``dynamicsymbols(f'u1_{joint.name}')``.
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
    joint_axis : Vector, optional
        The rotation as well as translation axis. Note that the components of
        this axis are the same in the parent_interframe and child_interframe.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Body
        The joint's parent body.
    child : Body
        The joint's child body.
    rotation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the rotation angle.
    translation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the translation distance.
    rotation_speed : dynamicsymbol
        Generalized speed corresponding to the angular velocity.
    translation_speed : dynamicsymbol
        Generalized speed corresponding to the translation velocity.
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
    joint_axis : Vector
        The axis of rotation and translation.

    Examples
    =========

    A single cylindrical joint is created between two bodies and has the
    following basic attributes:

    >>> from sympy.physics.mechanics import Body, CylindricalJoint
    >>> parent = Body('P')
    >>> parent
    P
    >>> child = Body('C')
    >>> child
    C
    >>> joint = CylindricalJoint('PC', parent, child)
    >>> joint
    CylindricalJoint: PC  parent: P  child: C
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
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)]])
    >>> joint.child.frame.ang_vel_in(joint.parent.frame)
    u0_PC(t)*P_frame.x
    >>> joint.child.frame.dcm(joint.parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.x
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.x

    To further demonstrate the use of the cylindrical joint, the kinematics of
    two cylindrical joints perpendicular to each other can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Body, CylindricalJoint
    >>> r, l, w = symbols('r l w')

    First create bodies to represent the fixed floor with a fixed pole on it.
    The second body represents a freely moving tube around that pole. The third
    body represents a solid flag freely translating along and rotating around
    the Y axis of the tube.

    >>> floor = Body('floor')
    >>> tube = Body('tube')
    >>> flag = Body('flag')

    The first joint will connect the first tube to the floor with it translating
    along and rotating around the Z axis of both bodies.

    >>> floor_joint = CylindricalJoint('C1', floor, tube, joint_axis=floor.z)

    The second joint will connect the tube perpendicular to the flag along the Y
    axis of both the tube and the flag, with the joint located at a distance
    ``r`` from the tube's center of mass and a combination of the distances
    ``l`` and ``w`` from the flag's center of mass.

    >>> flag_joint = CylindricalJoint('C2', tube, flag,
    ...                               parent_point=r * tube.y,
    ...                               child_point=-w * flag.y + l * flag.z,
    ...                               joint_axis=tube.y)

    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of both the body and the
    flag relative to the floor are found:

    >>> tube.dcm(floor)
    Matrix([
    [ cos(q0_C1(t)), sin(q0_C1(t)), 0],
    [-sin(q0_C1(t)), cos(q0_C1(t)), 0],
    [             0,             0, 1]])
    >>> flag.dcm(floor)
    Matrix([
    [cos(q0_C1(t))*cos(q0_C2(t)), sin(q0_C1(t))*cos(q0_C2(t)), -sin(q0_C2(t))],
    [             -sin(q0_C1(t)),               cos(q0_C1(t)),              0],
    [sin(q0_C2(t))*cos(q0_C1(t)), sin(q0_C1(t))*sin(q0_C2(t)),  cos(q0_C2(t))]])

    The position of the flag's center of mass is found with:

    >>> flag.masscenter.pos_from(floor.masscenter)
    q1_C1(t)*floor_frame.z + (r + q1_C2(t))*tube_frame.y + w*flag_frame.y - l*flag_frame.z

    The angular velocities of the two tubes can be computed with respect to the
    floor.

    >>> tube.ang_vel_in(floor)
    u0_C1(t)*floor_frame.z
    >>> flag.ang_vel_in(floor)
    u0_C1(t)*floor_frame.z + u0_C2(t)*tube_frame.y

    Finally, the linear velocities of the two tube centers of mass can be
    computed with respect to the floor, while expressed in the tube's frame.

    >>> tube.masscenter.vel(floor.frame).to_matrix(tube.frame)
    Matrix([
    [       0],
    [       0],
    [u1_C1(t)]])
    >>> flag.masscenter.vel(floor.frame).to_matrix(tube.frame).simplify()
    Matrix([
    [-l*u0_C2(t)*cos(q0_C2(t)) - r*u0_C1(t) - w*u0_C1(t) - q1_C2(t)*u0_C1(t)],
    [                    -l*u0_C1(t)*sin(q0_C2(t)) + Derivative(q1_C2(t), t)],
    [                                    l*u0_C2(t)*sin(q0_C2(t)) + u1_C1(t)]])

    """

    def __init__(self, name, parent, child, rotation_coordinate=None, translation_coordinate=None, rotation_speed=None, translation_speed=None, parent_point=None, child_point=None, parent_interframe=None, child_interframe=None, joint_axis=None):
        self._joint_axis = joint_axis
        coordinates = (rotation_coordinate, translation_coordinate)
        speeds = (rotation_speed, translation_speed)
        super().__init__(name, parent, child, coordinates, speeds, parent_point, child_point, parent_interframe=parent_interframe, child_interframe=child_interframe)

    def __str__(self):
        return f'CylindricalJoint: {self.name}  parent: {self.parent}  child: {self.child}'

    @property
    def joint_axis(self):
        """Axis about and along which the rotation and translation occurs."""
        return self._joint_axis

    @property
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    def translation_coordinate(self):
        """Generalized coordinate corresponding to the translation distance."""
        return self.coordinates[1]

    @property
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    def translation_speed(self):
        """Generalized speed corresponding to the translation velocity."""
        return self.speeds[1]

    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 2, 'q')

    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, 2, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(self.parent_interframe, self.joint_axis, self.rotation_coordinate)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, self.rotation_speed * self.joint_axis.normalize())

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, self.translation_coordinate * self.joint_axis.normalize())
        self.parent_point.set_vel(self.parent.frame, 0)
        self.child_point.set_vel(self.child.frame, 0)
        self.child_point.set_vel(self.parent.frame, self.translation_speed * self.joint_axis.normalize())
        self.child.masscenter.v2pt_theory(self.child_point, self.parent.frame, self.child_interframe)