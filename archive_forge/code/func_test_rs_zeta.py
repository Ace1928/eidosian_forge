from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_rs_zeta():
    mp.dps = 15
    assert zeta(0.5 + 100000j).ae(1.0730320148577532 + 5.780848544363504j)
    assert zeta(0.75 + 100000j).ae(1.8378523372518738 + 1.9988492668661146j)
    assert zeta(0.5 + 1000000j, derivative=3).ae(1647.7744105852676 - 1423.1270943036623j)
    assert zeta(1 + 1000000j, derivative=3).ae(3.4085866124523583 - 18.179184721525946j)
    assert zeta(1 + 1000000j, derivative=1).ae(-0.10423479366985453 - 0.7472899280335905j)
    assert zeta(0.5 - 1000000j, derivative=1).ae(11.636804066002522 + 17.127254072212995j)

    def ae(x, y, tol=1e-06):
        return abs(x - y) < tol * abs(y)
    assert ae(fp.zeta(0.5 - 100000j), 1.0730320148577532 - 5.780848544363504j)
    assert ae(fp.zeta(0.75 - 100000j), 1.8378523372518738 - 1.9988492668661146j)
    assert ae(fp.zeta(0.5 + 1000000j), 0.0760890697382271 + 2.805102101019299j)
    assert ae(fp.zeta(0.5 + 1000000j, derivative=1), 11.636804066002522 - 17.127254072212995j)
    assert ae(fp.zeta(1 + 1000000j), 0.9473872625104789 + 0.5942199931209183j)
    assert ae(fp.zeta(1 + 1000000j, derivative=1), -0.10423479366985453 - 0.7472899280335905j)
    assert ae(fp.zeta(0.5 + 100000j, derivative=1), 10.766962036817482 - 30.927052821059966j)
    assert ae(fp.zeta(0.5 + 100000j, derivative=2), -119.40515625740538 + 217.1478063114183j)
    assert ae(fp.zeta(0.5 + 100000j, derivative=3), 1129.755028262846 - 1685.473689516969j)
    assert ae(fp.zeta(0.5 + 100000j, derivative=4), -10407.160819314959 + 13777.786698628044j)
    assert ae(fp.zeta(0.75 + 100000j, derivative=1), -0.4174227669959432 - 6.445381627504996j)
    assert ae(fp.zeta(0.75 + 100000j, derivative=2), -9.214314279161977 + 35.072907953379676j)
    assert ae(fp.zeta(0.75 + 100000j, derivative=3), 110.61331857820103 - 236.8784713051813j)
    assert ae(fp.zeta(0.75 + 100000j, derivative=4), -1054.3342758985593 + 1769.9177890161595j)