from sympy.core.backend import (Symbol, symbols, sin, cos, Matrix, zeros,
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.mechanics import inertia, Body
from sympy.testing.pytest import raises
def test_particle_body():
    particle_masscenter = Point('particle_masscenter')
    particle_mass = Symbol('particle_mass')
    particle_frame = ReferenceFrame('particle_frame')
    particle_body = Body('particle_body', particle_masscenter, particle_mass, particle_frame)
    com = particle_body.masscenter
    frame = particle_body.frame
    particle_masscenter.set_vel(particle_frame, 0)
    assert com.vel(frame) == particle_masscenter.vel(frame)
    assert com.pos_from(com) == particle_masscenter.pos_from(com)
    assert particle_body.mass == particle_mass
    assert not hasattr(particle_body, '_inertia')
    assert hasattr(particle_body, 'frame')
    assert hasattr(particle_body, 'masscenter')
    assert hasattr(particle_body, 'mass')
    assert particle_body.inertia == (Dyadic(0), particle_body.masscenter)
    assert particle_body.central_inertia == Dyadic(0)
    assert not particle_body.is_rigidbody
    particle_body.central_inertia = inertia(particle_frame, 1, 1, 1)
    assert particle_body.central_inertia == inertia(particle_frame, 1, 1, 1)
    assert particle_body.is_rigidbody
    particle_body = Body('particle_body', mass=particle_mass)
    assert not particle_body.is_rigidbody
    point = particle_body.masscenter.locatenew('point', particle_body.x)
    point_inertia = particle_mass * inertia(particle_body.frame, 0, 1, 1)
    particle_body.inertia = (point_inertia, point)
    assert particle_body.inertia == (point_inertia, point)
    assert particle_body.central_inertia == Dyadic(0)
    assert particle_body.is_rigidbody