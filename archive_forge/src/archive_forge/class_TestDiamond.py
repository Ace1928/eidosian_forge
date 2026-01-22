import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
class TestDiamond(BaseTestByIR):
    refprune_bitmask = llvm.RefPruneSubpasses.DIAMOND
    per_diamond_1 = '\ndefine void @main(i8* %ptr) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br label %bb_B\nbb_B:\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_per_diamond_1(self):
        mod, stats = self.check(self.per_diamond_1)
        self.assertEqual(stats.diamond, 2)
    per_diamond_2 = '\ndefine void @main(i8* %ptr, i1 %cond) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    br label %bb_D\nbb_C:\n    br label %bb_D\nbb_D:\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_per_diamond_2(self):
        mod, stats = self.check(self.per_diamond_2)
        self.assertEqual(stats.diamond, 2)
    per_diamond_3 = '\ndefine void @main(i8* %ptr, i1 %cond) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    br label %bb_D\nbb_C:\n    call void @NRT_decref(i8* %ptr)  ; reject because of decref in diamond\n    br label %bb_D\nbb_D:\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_per_diamond_3(self):
        mod, stats = self.check(self.per_diamond_3)
        self.assertEqual(stats.diamond, 0)
    per_diamond_4 = '\ndefine void @main(i8* %ptr, i1 %cond) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    call void @NRT_incref(i8* %ptr)     ; extra incref will not affect prune\n    br label %bb_D\nbb_C:\n    br label %bb_D\nbb_D:\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_per_diamond_4(self):
        mod, stats = self.check(self.per_diamond_4)
        self.assertEqual(stats.diamond, 2)
    per_diamond_5 = '\ndefine void @main(i8* %ptr, i1 %cond) {\nbb_A:\n    call void @NRT_incref(i8* %ptr)\n    call void @NRT_incref(i8* %ptr)\n    br i1 %cond, label %bb_B, label %bb_C\nbb_B:\n    br label %bb_D\nbb_C:\n    br label %bb_D\nbb_D:\n    call void @NRT_decref(i8* %ptr)\n    call void @NRT_decref(i8* %ptr)\n    ret void\n}\n'

    def test_per_diamond_5(self):
        mod, stats = self.check(self.per_diamond_5)
        self.assertEqual(stats.diamond, 4)