import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
class TestPerBB(BaseTestByIR):
    refprune_bitmask = llvm.RefPruneSubpasses.PER_BB
    per_bb_ir_1 = '\ndefine void @main(i8* %ptr) {\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_per_bb_1(self):
        mod, stats = self.check(self.per_bb_ir_1)
        self.assertEqual(stats.basicblock, 2)
    per_bb_ir_2 = '\ndefine void @main(i8* %ptr) {\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_per_bb_2(self):
        mod, stats = self.check(self.per_bb_ir_2)
        self.assertEqual(stats.basicblock, 4)
        self.assertIn('call void @NRT_incref(i8* %ptr)', str(mod))
    per_bb_ir_3 = '\ndefine void @main(i8* %ptr, i8* %other) {\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    call void @NRT_decref(i8* %other)\n    ret void\n}\n'

    def test_per_bb_3(self):
        mod, stats = self.check(self.per_bb_ir_3)
        self.assertEqual(stats.basicblock, 2)
        self.assertIn('call void @NRT_decref(i8* %other)', str(mod))
    per_bb_ir_4 = '\n; reordered\ndefine void @main(i8* %ptr, i8* %other) {\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    call void @NRT_decref(i8* %other)\n    call void @NRT_incref(i8* %ptr)\n    ret void\n}\n'

    def test_per_bb_4(self):
        mod, stats = self.check(self.per_bb_ir_4)
        self.assertEqual(stats.basicblock, 4)
        self.assertIn('call void @NRT_decref(i8* %other)', str(mod))