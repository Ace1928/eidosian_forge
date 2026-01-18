from sympy.polys.rings import vring
from sympy.polys.fields import vfield
from sympy.polys.domains import QQ
def test_vring():
    ns = {'vring': vring, 'QQ': QQ}
    exec('R = vring("r", QQ)', ns)
    exec('assert r == R.gens[0]', ns)
    exec('R = vring("rb rbb rcc rzz _rx", QQ)', ns)
    exec('assert rb == R.gens[0]', ns)
    exec('assert rbb == R.gens[1]', ns)
    exec('assert rcc == R.gens[2]', ns)
    exec('assert rzz == R.gens[3]', ns)
    exec('assert _rx == R.gens[4]', ns)
    exec('R = vring(["rd", "re", "rfg"], QQ)', ns)
    exec('assert rd == R.gens[0]', ns)
    exec('assert re == R.gens[1]', ns)
    exec('assert rfg == R.gens[2]', ns)