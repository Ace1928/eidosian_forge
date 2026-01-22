from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
from sympy.physics.mechanics.method import _Methods
from sympy.core.backend import Matrix
def _convert_bodies(self):
    bodylist = []
    for body in self.bodies:
        if body.is_rigidbody:
            rb = RigidBody(body.name, body.masscenter, body.frame, body.mass, (body.central_inertia, body.masscenter))
            rb.potential_energy = body.potential_energy
            bodylist.append(rb)
        else:
            part = Particle(body.name, body.masscenter, body.mass)
            part.potential_energy = body.potential_energy
            bodylist.append(part)
    return bodylist