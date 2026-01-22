import subprocess
import sys
import os
import time

This module is just for testing concepts. It should be erased later on.

Experiments:

// gdb -p 4957
// call dlopen("/home/fabioz/Desktop/dev/PyDev.Debugger/pydevd_attach_to_process/linux/attach_linux.so", 2)
// call dlsym($1, "hello")
// call hello()


// call open("/home/fabioz/Desktop/dev/PyDev.Debugger/pydevd_attach_to_process/linux/attach_linux.so", 2)
// call mmap(0, 6672, 1 | 2 | 4, 1, 3 , 0)
// add-symbol-file
// cat /proc/pid/maps

// call dlopen("/home/fabioz/Desktop/dev/PyDev.Debugger/pydevd_attach_to_process/linux/attach_linux.so", 1|8)
// call dlsym($1, "hello")
// call hello()
