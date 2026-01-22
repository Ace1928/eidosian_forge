from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
class PlanarJoint(Joint):
    """Planar Joint.

    .. image:: PlanarJoint.svg
        :align: center
        :width: 800

    Explanation
    ===========

    A planar joint is defined such that the child body translates over a fixed
    plane of the parent body as well as rotate about the rotation axis, which
    is perpendicular to that plane. The origin of this plane is the
    ``parent_point`` and the plane is spanned by two nonparallel planar vectors.
    The location of the ``child_point`` is based on the planar vectors
    ($\\vec{v}_1$, $\\vec{v}_2$) and generalized coordinates ($q_1$, $q_2$),
    i.e. $\\vec{r} = q_1 \\hat{v}_1 + q_2 \\hat{v}_2$. The direction cosine
    matrix between the ``child_interframe`` and ``parent_interframe`` is formed
    using a simple rotation ($q_0$) about the rotation axis.

    In order to simplify the definition of the ``PlanarJoint``, the
    ``rotation_axis`` and ``planar_vectors`` are set to be the unit vectors of
    the ``parent_interframe`` according to the table below. This ensures that
    you can only define these vectors by creating a separate frame and supplying
    that as the interframe. If you however would only like to supply the normals
    of the plane with respect to the parent and child bodies, then you can also
    supply those to the ``parent_interframe`` and ``child_interframe``
    arguments. An example of both of these cases is in the examples section
    below and the page on the joints framework provides a more detailed
    explanation of the intermediate frames.

    .. list-table::

        * - ``rotation_axis``
          - ``parent_interframe.x``
        * - ``planar_vectors[0]``
          - ``parent_interframe.y``
        * - ``planar_vectors[1]``
          - ``parent_interframe.z``

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
    planar_coordinates : iterable of dynamicsymbols, optional
        Two generalized coordinates used for the planar translation. The default
        value is ``dynamicsymbols(f'q1_{joint.name} q2_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    planar_speeds : dynamicsymbols, optional
        Two generalized speeds used for the planar translation velocity. The
        default value is ``dynamicsymbols(f'u1_{joint.name} u2_{joint.name}')``.
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
    planar_coordinates : Matrix
        Two generalized coordinates used for the planar translation.
    rotation_speed : dynamicsymbol
        Generalized speed corresponding to the angular velocity.
    planar_speeds : Matrix
        Two generalized speeds used for the planar translation velocity.
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
    rotation_axis : Vector
        The axis about which the rotation occurs.
    planar_vectors : list
        The vectors that describe the planar translation directions.

    Examples
    =========

    A single planar joint is created between two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import Body, PlanarJoint
    >>> parent = Body('P')
    >>> parent
    P
    >>> child = Body('C')
    >>> child
    C
    >>> joint = PlanarJoint('PC', parent, child)
    >>> joint
    PlanarJoint: PC  parent: P  child: C
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
    >>> joint.rotation_axis
    P_frame.x
    >>> joint.planar_vectors
    [P_frame.y, P_frame.z]
    >>> joint.rotation_coordinate
    q0_PC(t)
    >>> joint.planar_coordinates
    Matrix([
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.rotation_speed
    u0_PC(t)
    >>> joint.planar_speeds
    Matrix([
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> joint.child.frame.ang_vel_in(joint.parent.frame)
    u0_PC(t)*P_frame.x
    >>> joint.child.frame.dcm(joint.parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.y + q2_PC(t)*P_frame.z
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.y + u2_PC(t)*P_frame.z

    To further demonstrate the use of the planar joint, the kinematics of a
    block sliding on a slope, can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import PlanarJoint, Body, ReferenceFrame
    >>> a, d, h = symbols('a d h')

    First create bodies to represent the slope and the block.

    >>> ground = Body('G')
    >>> block = Body('B')

    To define the slope you can either define the plane by specifying the
    ``planar_vectors`` or/and the ``rotation_axis``. However it is advisable to
    create a rotated intermediate frame, so that the ``parent_vectors`` and
    ``rotation_axis`` will be the unit vectors of this intermediate frame.

    >>> slope = ReferenceFrame('A')
    >>> slope.orient_axis(ground.frame, ground.y, a)

    The planar joint can be created using these bodies and intermediate frame.
    We can specify the origin of the slope to be ``d`` above the slope's center
    of mass and the block's center of mass to be a distance ``h`` above the
    slope's surface. Note that we can specify the normal of the plane using the
    rotation axis argument.

    >>> joint = PlanarJoint('PC', ground, block, parent_point=d * ground.x,
    ...                     child_point=-h * block.x, parent_interframe=slope)

    Once the joint is established the kinematics of the bodies can be accessed.
    First the ``rotation_axis``, which is normal to the plane and the
    ``plane_vectors``, can be found.

    >>> joint.rotation_axis
    A.x
    >>> joint.planar_vectors
    [A.y, A.z]

    The direction cosine matrix of the block with respect to the ground can be
    found with:

    >>> block.dcm(ground)
    Matrix([
    [              cos(a),              0,              -sin(a)],
    [sin(a)*sin(q0_PC(t)),  cos(q0_PC(t)), sin(q0_PC(t))*cos(a)],
    [sin(a)*cos(q0_PC(t)), -sin(q0_PC(t)), cos(a)*cos(q0_PC(t))]])

    The angular velocity of the block can be computed with respect to the
    ground.

    >>> block.ang_vel_in(ground)
    u0_PC(t)*A.x

    The position of the block's center of mass can be found with:

    >>> block.masscenter.pos_from(ground.masscenter)
    d*G_frame.x + h*B_frame.x + q1_PC(t)*A.y + q2_PC(t)*A.z

    Finally, the linear velocity of the block's center of mass can be
    computed with respect to the ground.

    >>> block.masscenter.vel(ground.frame)
    u1_PC(t)*A.y + u2_PC(t)*A.z

    In some cases it could be your preference to only define the normals of the
    plane with respect to both bodies. This can most easily be done by supplying
    vectors to the ``interframe`` arguments. What will happen in this case is
    that an interframe will be created with its ``x`` axis aligned with the
    provided vector. For a further explanation of how this is done see the notes
    of the ``Joint`` class. In the code below, the above example (with the block
    on the slope) is recreated by supplying vectors to the interframe arguments.
    Note that the previously described option is however more computationally
    efficient, because the algorithm now has to compute the rotation angle
    between the provided vector and the 'x' axis.

    >>> from sympy import symbols, cos, sin
    >>> from sympy.physics.mechanics import PlanarJoint, Body
    >>> a, d, h = symbols('a d h')
    >>> ground = Body('G')
    >>> block = Body('B')
    >>> joint = PlanarJoint(
    ...     'PC', ground, block, parent_point=d * ground.x,
    ...     child_point=-h * block.x, child_interframe=block.x,
    ...     parent_interframe=cos(a) * ground.x + sin(a) * ground.z)
    >>> block.dcm(ground).simplify()
    Matrix([
    [               cos(a),              0,               sin(a)],
    [-sin(a)*sin(q0_PC(t)),  cos(q0_PC(t)), sin(q0_PC(t))*cos(a)],
    [-sin(a)*cos(q0_PC(t)), -sin(q0_PC(t)), cos(a)*cos(q0_PC(t))]])

    """

    def __init__(self, name, parent, child, rotation_coordinate=None, planar_coordinates=None, rotation_speed=None, planar_speeds=None, parent_point=None, child_point=None, parent_interframe=None, child_interframe=None):
        coordinates = (rotation_coordinate, planar_coordinates)
        speeds = (rotation_speed, planar_speeds)
        super().__init__(name, parent, child, coordinates, speeds, parent_point, child_point, parent_interframe=parent_interframe, child_interframe=child_interframe)

    def __str__(self):
        return f'PlanarJoint: {self.name}  parent: {self.parent}  child: {self.child}'

    @property
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    def planar_coordinates(self):
        """Two generalized coordinates used for the planar translation."""
        return self.coordinates[1:, 0]

    @property
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    def planar_speeds(self):
        """Two generalized speeds used for the planar translation velocity."""
        return self.speeds[1:, 0]

    @property
    def rotation_axis(self):
        """The axis about which the rotation occurs."""
        return self.parent_interframe.x

    @property
    def planar_vectors(self):
        """The vectors that describe the planar translation directions."""
        return [self.parent_interframe.y, self.parent_interframe.z]

    def _generate_coordinates(self, coordinates):
        rotation_speed = self._fill_coordinate_list(coordinates[0], 1, 'q', number_single=True)
        planar_speeds = self._fill_coordinate_list(coordinates[1], 2, 'q', 1)
        return rotation_speed.col_join(planar_speeds)

    def _generate_speeds(self, speeds):
        rotation_speed = self._fill_coordinate_list(speeds[0], 1, 'u', number_single=True)
        planar_speeds = self._fill_coordinate_list(speeds[1], 2, 'u', 1)
        return rotation_speed.col_join(planar_speeds)

    def _orient_frames(self):
        self.child_interframe.orient_axis(self.parent_interframe, self.rotation_axis, self.rotation_coordinate)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, self.rotation_speed * self.rotation_axis)

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, self.planar_coordinates[0] * self.planar_vectors[0] + self.planar_coordinates[1] * self.planar_vectors[1])
        self.parent_point.set_vel(self.parent_interframe, 0)
        self.child_point.set_vel(self.child_interframe, 0)
        self.child_point.set_vel(self.parent.frame, self.planar_speeds[0] * self.planar_vectors[0] + self.planar_speeds[1] * self.planar_vectors[1])
        self.child.masscenter.v2pt_theory(self.child_point, self.parent.frame, self.child.frame)