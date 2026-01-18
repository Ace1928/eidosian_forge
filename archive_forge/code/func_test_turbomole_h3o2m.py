from math import radians, sin, cos
from ase import Atoms
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton, BFGS
from ase.calculators.turbomole import Turbomole
def test_turbomole_h3o2m():
    doo = 2.74
    doht = 0.957
    doh = 0.977
    angle = radians(104.5)
    initial = Atoms('HOHOH', positions=[(-sin(angle) * doht, 0.0, cos(angle) * doht), (0.0, 0.0, 0.0), (0.0, 0.0, doh), (0.0, 0.0, doo), (sin(angle) * doht, 0.0, doo - cos(angle) * doht)])
    final = Atoms('HOHOH', positions=[(-sin(angle) * doht, 0.0, cos(angle) * doht), (0.0, 0.0, 0.0), (0.0, 0.0, doo - doh), (0.0, 0.0, doo), (sin(angle) * doht, 0.0, doo - cos(angle) * doht)])
    images = [initial.copy()]
    for i in range(3):
        images.append(initial.copy())
    images.append(final.copy())
    neb = NEB(images, climb=True)
    define_str = '\n\na coord\n\n*\nno\nb all 3-21g hondo\n*\neht\n\n-1\nno\ns\n*\n\ndft\non\nfunc pwlda\n\n\nscf\niter\n300\n\n*'
    constraint = FixAtoms(indices=[1, 3])
    for image in images:
        image.calc = Turbomole(define_str=define_str)
        image.set_constraint(constraint)
    with QuasiNewton(images[0]) as dyn1:
        dyn1.run(fmax=0.1)
    with QuasiNewton(images[-1]) as dyn2:
        dyn2.run(fmax=0.1)
    neb.interpolate()
    for image in images:
        print(image.get_distance(1, 2), image.get_potential_energy())
    with BFGS(neb, trajectory='turbomole_h3o2m.traj') as dyn:
        dyn.run(fmax=0.1)
    for image in images:
        print(image.get_distance(1, 2), image.get_potential_energy())