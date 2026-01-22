import collections
import time
import unittest
import os
import pygame
class EventModuleArgsTest(unittest.TestCase):

    def setUp(self):
        pygame.display.init()
        pygame.event.clear()

    def tearDown(self):
        pygame.display.quit()

    def test_get(self):
        pygame.event.get()
        pygame.event.get(None)
        pygame.event.get(None, True)
        pygame.event.get(pump=False)
        pygame.event.get(pump=True)
        pygame.event.get(eventtype=None)
        pygame.event.get(eventtype=[pygame.KEYUP, pygame.KEYDOWN])
        pygame.event.get(eventtype=pygame.USEREVENT, pump=False)
        self.assertRaises(ValueError, pygame.event.get, 65536)
        self.assertRaises(TypeError, pygame.event.get, 1 + 2j)
        self.assertRaises(TypeError, pygame.event.get, 'foo')

    def test_clear(self):
        pygame.event.clear()
        pygame.event.clear(None)
        pygame.event.clear(None, True)
        pygame.event.clear(pump=False)
        pygame.event.clear(pump=True)
        pygame.event.clear(eventtype=None)
        pygame.event.clear(eventtype=[pygame.KEYUP, pygame.KEYDOWN])
        pygame.event.clear(eventtype=pygame.USEREVENT, pump=False)
        self.assertRaises(ValueError, pygame.event.clear, 17825791)
        self.assertRaises(TypeError, pygame.event.get, ['a', 'b', 'c'])

    def test_peek(self):
        pygame.event.peek()
        pygame.event.peek(None)
        pygame.event.peek(None, True)
        pygame.event.peek(pump=False)
        pygame.event.peek(pump=True)
        pygame.event.peek(eventtype=None)
        pygame.event.peek(eventtype=[pygame.KEYUP, pygame.KEYDOWN])
        pygame.event.peek(eventtype=pygame.USEREVENT, pump=False)

        class Foo:
            pass
        self.assertRaises(ValueError, pygame.event.peek, -1)
        self.assertRaises(ValueError, pygame.event.peek, [-10])
        self.assertRaises(TypeError, pygame.event.peek, Foo())