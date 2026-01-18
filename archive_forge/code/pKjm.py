import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomTriangles,
)
from panda3d.core import RenderModeAttrib


def construct_cube():
    """
    Construct a cube using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a cube with detailed vertex and color definitions, using structured arrays for optimal data management.
    """

    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "cube_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy arrays for structured data management
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Bottom
            [4, 5, 6],
            [4, 6, 7],  # Top
            [4, 5, 1],
            [4, 1, 0],  # Front
            [6, 7, 3],
            [6, 3, 2],  # Back
            [4, 0, 3],
            [4, 3, 7],  # Left
            [5, 1, 2],
            [5, 2, 6],  # Right
        ],
        dtype=np.int32,
    )

    # Create triangle primitives with static usage hint
    triangles = GeomTriangles(Geom.UHStatic)

    # Add triangles to the primitive
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("cube_geom_node")
    geometry_node.addGeom(geometry)


def construct_prism_with_vertex_data():
    """
    Construct a prism using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "prism_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],  # Base triangle
            [0, 0, 1],
            [1, 0, 1],
            [0.5, 1, 1],  # Top triangle parallel to base
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],  # Base colors
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],  # Top colors
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],  # Base and top
            [0, 3, 4],
            [0, 4, 1],  # Sides
            [1, 4, 5],
            [1, 5, 2],
            [2, 5, 3],
            [2, 3, 0],
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("prism_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_triangle_sheet_with_vertex_data():
    """
    Construct a triangle sheet using vertex data and geom nodes, focusing on detailed vertex and color management.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData(
        "triangle_sheet_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)

    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array([[0, 1, 2]], dtype=np.int32)

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("triangle_sheet_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_square_sheet_with_vertex_data():
    """
    Construct a square sheet using vertex data and geom nodes, with a focus on structured data management.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData(
        "square_sheet_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

    colors = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]], dtype=np.float32
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("square_sheet_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_circle_sheet_with_vertex_data():
    """
    Construct a circle sheet using vertex data and geom nodes, employing structured arrays for vertex and color data.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData(
        "circle_sheet_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    num_segments = 32
    radius = 1.0
    angle_increment = 2 * np.pi / num_segments
    vertices = np.array(
        [
            [
                np.cos(i * angle_increment) * radius,
                np.sin(i * angle_increment) * radius,
                0,
            ]
            for i in range(num_segments)
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [np.cos(i * angle_increment), np.sin(i * angle_increment), 0.5, 1]
            for i in range(num_segments)
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [[i, (i + 1) % num_segments, num_segments] for i in range(num_segments)],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("circle_sheet_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_triangle_prism_with_vertex_data():
    """
    Construct a triangle prism using vertex data and geom nodes, with detailed management of vertex and color data.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData(
        "triangle_prism_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],  # Base triangle
            [0, 0, 1],
            [1, 0, 1],
            [0.5, 1, 1],  # Top triangle parallel to base
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],  # Base colors
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],  # Top colors
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],  # Base and top
            [0, 3, 4],
            [0, 4, 1],  # Sides
            [1, 4, 5],
            [1, 5, 2],
            [2, 5, 3],
            [2, 3, 0],
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("triangle_prism_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_pyramid():
    """
    Construct a pyramid using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "pyramid_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array(
        [
            [0, 0, 1],  # Apex
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],  # Base vertices
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],  # Red for Apex
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 1],  # Base colors
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],  # Sides
            [1, 2, 3],
            [1, 3, 4],  # Base
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("pyramid_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_rectangular_prism():
    """
    Construct a rectangular prism using vertex data and geom nodes, with detailed vertex and color definitions.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData(
        "rectangular_prism_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],  # Bottom vertices
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],  # Top vertices
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],  # Bottom colors
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 1],  # Top colors
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Bottom
            [4, 5, 6],
            [4, 6, 7],  # Top
            [0, 4, 5],
            [0, 5, 1],  # Front
            [2, 6, 7],
            [2, 7, 3],  # Back
            [0, 4, 7],
            [0, 7, 3],  # Left
            [1, 5, 6],
            [1, 6, 2],  # Right
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("rectangular_prism_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_cuboid():
    """
    Construct a cuboid using vertex data and geom nodes, ensuring detailed vertex and color management.
    """
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData(
        "cuboid_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array(
        [
            [-1, -0.5, -1],
            [1, -0.5, -1],
            [1, 0.5, -1],
            [-1, 0.5, -1],  # Bottom vertices
            [-1, -0.5, 1],
            [1, -0.5, 1],
            [1, 0.5, 1],
            [-1, 0.5, 1],  # Top vertices
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],  # Bottom colors
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 1],  # Top colors
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Bottom
            [4, 5, 6],
            [4, 6, 7],  # Top
            [0, 4, 5],
            [0, 5, 1],  # Front
            [2, 6, 7],
            [2, 7, 3],  # Back
            [0, 4, 7],
            [0, 7, 3],  # Left
            [1, 5, 6],
            [1, 6, 2],  # Right
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("cuboid_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_rhomboid():
    """
    Construct a rhomboid using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a rhomboid with detailed vertex and color definitions, using structured arrays for optimal data management.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "rhomboid_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy arrays for structured data management
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1.5, 1, 0],
            [0.5, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1.5, 1, 1],
            [0.5, 1, 1],
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 7],
            [0, 7, 3],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int32,
    )

    # Create triangle primitives with static usage hint
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("rhomboid_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_parallelepiped():
    """
    Construct a parallelepiped using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a parallelepiped with detailed vertex and color definitions, using structured arrays for optimal data management.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "parallelepiped_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy arrays for structured data management
    vertices = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [0, 1, 0],
            [0.5, 0, 1],
            [2.5, 0, 1],
            [2.5, 1, 1],
            [0.5, 1, 1],
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 7],
            [0, 7, 3],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int32,
    )

    # Create triangle primitives with static usage hint
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("parallelepiped_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_trapezoidal_prism():
    """
    Construct a trapezoidal prism using vertex data and geom nodes.
    This function meticulously constructs a trapezoidal prism with detailed vertex and color definitions,
    using structured arrays for optimal data management and efficiency.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "trapezoidal_prism_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy arrays for structured data management
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.85, 0.5, 0],
            [0.15, 0.5, 0],  # Bottom face
            [0, 0, 1],
            [1, 0, 1],
            [0.85, 0.5, 1],
            [0.15, 0.5, 1],  # Top face parallel to bottom
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],  # Bottom face colors
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 1],  # Top face colors
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Bottom face
            [4, 5, 6],
            [4, 6, 7],  # Top face
            [0, 4, 5],
            [0, 5, 1],  # Sides
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )

    # Create triangle primitives with static usage hint
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("trapezoidal_prism_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_trapezoidal_pyramid():
    """
    Construct a trapezoidal pyramid using vertex data and geom nodes.
    This function meticulously constructs a trapezoidal pyramid with detailed vertex and color definitions,
    using structured arrays for optimal data management and efficiency.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "trapezoidal_pyramid_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy arrays for structured data management
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.85, 0.5, 0],
            [0.15, 0.5, 0],  # Base vertices
            [0.5, 0.25, 1],  # Apex of the pyramid
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],  # Base colors
            [1, 0, 1, 1],  # Apex color
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],  # Sides
            [0, 1, 2],
            [0, 2, 3],  # Base
        ],
        dtype=np.int32,
    )

    # Create triangle primitives with static usage hint
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("trapezoidal_pyramid_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_conical_frustum():
    """
    Construct a conical frustum using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a conical frustum with detailed vertex and color definitions, using structured arrays for optimal data management.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "conical_frustum_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    r1 = 1.0  # Top radius
    r2 = 0.5  # Bottom radius
    h = 1.0  # Height

    # Define vertices and colors using numpy structured arrays
    # Assuming the frustum has a top radius 'r1', bottom radius 'r2', and height 'h'
    num_segments = 36  # Define the number of segments for the frustum
    angle_increment = 2 * np.pi / num_segments
    heights = np.array([0, h], dtype=np.float32)  # Bottom and top heights
    radii = np.array([r2, r1], dtype=np.float32)  # Bottom and top radii

    vertices = np.array(
        [
            [
                np.cos(i * angle_increment) * radii[j],
                np.sin(i * angle_increment) * radii[j],
                heights[j],
            ]
            for j in range(2)
            for i in range(num_segments)
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 1, 1, 1] if i % 2 == 0 else [0.5, 0.5, 0.5, 1]
            for _ in range(2)
            for i in range(num_segments)
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [[i, (i + 1) % num_segments, i + num_segments] for i in range(num_segments)]
        + [
            [i, i + num_segments, (i + 1) % num_segments + num_segments]
            for i in range(num_segments)
        ],
        dtype=np.int32,
    )

    # Create triangle primitives with static usage hint
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("conical_frustum_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_cylindrical_frustum():
    """
    Construct a cylindrical frustum using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a cylindrical frustum with detailed vertex and color definitions, using structured arrays for optimal data management.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "cylindrical_frustum_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    r1 = 1.0  # Top radius
    r2 = 0.5  # Bottom radius
    h = 1.0  # Height

    # Define vertices and colors using numpy structured arrays
    # Assuming the frustum has a top radius 'r1', bottom radius 'r2', and height 'h'
    num_segments = 36  # Define the number of segments for the frustum
    angle_increment = 2 * np.pi / num_segments
    heights = np.array([0, 1], dtype=np.float32) * h  # Bottom and top heights
    radii = np.array([r2, r1], dtype=np.float32)  # Bottom and top radii

    vertices = np.array(
        [
            [
                np.cos(i * angle_increment) * radii[j],
                np.sin(i * angle_increment) * radii[j],
                heights[j],
            ]
            for j in range(2)
            for i in range(num_segments)
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [1, 1, 1, 1] if i % 2 == 0 else [0.5, 0.5, 0.5, 1]
            for _ in range(2)
            for i in range(num_segments)
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            (
                [i, (i + 1) % num_segments, i + num_segments]
                if i % 2 == 0
                else [
                    (i + 1) % num_segments,
                    i + num_segments,
                    (i + 1) % num_segments + num_segments,
                ]
            )
            for i in range(num_segments)
        ]
        + [
            (
                [i, i + num_segments, (i + 1) % num_segments + num_segments]
                if i % 2 == 0
                else [
                    i + num_segments,
                    (i + 1) % num_segments + num_segments,
                    (i + 1) % num_segments,
                ]
            )
            for i in range(num_segments)
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("cylindrical_frustum_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_cylinder(num_segments=32, height=1.0, radius=1.0):
    """
    Construct a cylinder using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "cylinder_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.zeros((num_segments * 2, 3), dtype=np.float32)
    colors = np.zeros((num_segments * 2, 4), dtype=np.float32)
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices[i] = [x, y, 0]  # Bottom circle
        vertices[i + num_segments] = [x, y, height]  # Top circle
        colors[i] = [1, 0, 0, 1]  # Red at bottom
        colors[i + num_segments] = [0, 0, 1, 1]  # Blue at top

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.zeros((num_segments * 2, 3), dtype=np.int32)
    for i in range(num_segments):
        triangle_indices[i] = [i, (i + 1) % num_segments, i + num_segments]
        triangle_indices[i + num_segments] = [
            (i + 1) % num_segments,
            i + num_segments,
            (i + 1) % num_segments + num_segments,
        ]

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("cylinder_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_cone(num_segments=32, height=1.0, radius=1.0):
    """
    Construct a cone using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "cone_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.zeros((num_segments + 1, 3), dtype=np.float32)
    colors = np.zeros((num_segments + 1, 4), dtype=np.float32)
    vertices[0] = [0, 0, height]  # Apex of the cone
    colors[0] = [1, 0, 0, 1]  # Red at apex
    for i in range(1, num_segments + 1):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices[i] = [x, y, 0]  # Base circle
        colors[i] = [0, 1, 0, 1]  # Green at base

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.zeros((num_segments, 3), dtype=np.int32)
    for i in range(num_segments):
        triangle_indices[i] = [0, i + 1, (i + 1) % num_segments + 1]

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("cone_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_sphere(num_segments=32, num_rings=16, radius=1.0):
    """
    Construct a sphere using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "sphere_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.zeros((num_segments * (num_rings - 1) + 2, 3), dtype=np.float32)
    colors = np.zeros((num_segments * (num_rings - 1) + 2, 4), dtype=np.float32)
    vertices[0] = [0, 0, -radius]  # Bottom pole
    colors[0] = [1, 0, 0, 1]  # Red at bottom
    vertices[-1] = [0, 0, radius]  # Top pole
    colors[-1] = [0, 0, 1, 1]  # Blue at top
    index = 1
    for j in range(1, num_rings):
        phi = np.pi * j / num_rings
        for i in range(num_segments):
            theta = 2 * np.pi * i / num_segments
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            vertices[index] = [x, y, z]
            colors[index] = [0, 1, 0, 1]  # Green elsewhere
            index += 1

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = []
    # Bottom cap
    for i in range(num_segments):
        triangle_indices.append([0, i + 1, (i + 1) % num_segments + 1])
    # Middle rings
    for j in range(1, num_rings - 1):
        for i in range(num_segments):
            current = (j - 1) * num_segments + i + 1
            next = current + num_segments
            triangle_indices.append([current, (current % num_segments) + 1, next])
            triangle_indices.append(
                [next, (current % num_segments) + 1, (next % num_segments) + 1]
            )
    # Top cap
    top_index = num_segments * (num_rings - 1) + 1
    for i in range(num_segments):
        current = (num_rings - 2) * num_segments + i + 1
        triangle_indices.append([current, (current % num_segments) + 1, top_index])

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("sphere_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node

    color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array([], dtype=np.int32)

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("torus_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_tetrahedron():
    """
    Construct a tetrahedron using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "tetrahedron_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    # Placeholder values, to be replaced with actual tetrahedron coordinates and colors
    vertices = np.array([], dtype=np.float32)
    colors = np.array([], dtype=np.float32)

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array([], dtype=np.int32)

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("tetrahedron_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_octahedron():
    """
    Construct an octahedron using vertex data and geom nodes, meticulously defining each vertex and its corresponding color.
    This function utilizes a structured array approach for optimal data management and efficiency.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "octahedron_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    # Placeholder values, to be replaced with actual octahedron coordinates and colors
    vertices = np.array([], dtype=np.float32)
    colors = np.array([], dtype=np.float32)

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array([], dtype=np.int32)

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("octahedron_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node

    return geometry_node
    geometry_node = GeomNode("octahedron_geom_node")
    # Define triangles using indices and numpy arrays
    triangle_indices = np.array([], dtype=np.int32)

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    for tri in triangle_indices:

        # Define triangles using indices and numpy arrays

        vertex_writer.addData3f(*vertex)
    # Add data to vertex and color writers
    vertices = np.array([], dtype=np.float32)
    # Define vertices and colors using numpy structured arrays


def construct_dodecahedron():
    """
    Construct a dodecahedron using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a dodecahedron with detailed vertex and color definitions, using structured arrays for optimal data management.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "dodecahedron_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array(
        [
            # Coordinates for the 20 vertices of a dodecahedron
            # Placeholder values, to be replaced with actual coordinates
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            # RGBA colors for each vertex
            # Placeholder values, to be replaced with actual color data
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define pentagons using indices and numpy arrays
    pentagon_indices = np.array(
        [
            # Indices for the 12 pentagon faces of a dodecahedron
            # Placeholder values, to be replaced with actual indices
        ],
        dtype=np.int32,
    )

    pentagons = GeomTriangles(Geom.UHStatic)
    for pent in pentagon_indices:
        pentagons.addVertices(*pent)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(pentagons)
    geometry_node = GeomNode("dodecahedron_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_icosahedron():
    """
    Construct an icosahedron using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs an icosahedron with detailed vertex and color definitions, using structured arrays for optimal data management.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "icosahedron_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array(
        [
            # Coordinates for the 12 vertices of an icosahedron
            # Placeholder values, to be replaced with actual coordinates
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            # RGBA colors for each vertex
            # Placeholder values, to be replaced with actual color data
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            # Indices for the 20 triangular faces of an icosahedron
            # Placeholder values, to be replaced with actual indices
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("icosahedron_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_geodesic_sphere():
    """
    Construct a geodesic sphere using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a geodesic sphere with detailed vertex and color definitions, using structured arrays for optimal data management.
    """
    vertex_format = GeomVertexFormat.getV3c4()  # 3D coordinates and RGBA colors
    vertex_data = GeomVertexData(
        "geodesic_sphere_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define vertices and colors using numpy structured arrays
    vertices = np.array(
        [
            # Coordinates for the vertices of a geodesic sphere
            # Placeholder values, to be replaced with actual coordinates
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            # RGBA colors for each vertex
            # Placeholder values, to be replaced with actual color data
        ],
        dtype=np.float32,
    )

    # Add data to vertex and color writers
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            # Indices for the triangular faces of a geodesic sphere
            # Placeholder values, to be replaced with actual indices
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("geodesic_sphere_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_spherical_frustum():
    """
    Construct a spherical frustum using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a spherical frustum with detailed vertex and color definitions, using structured arrays for optimal data management.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "spherical_frustum_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Define the parameters for the spherical frustum
    top_radius = 0.5  # Top radius of the frustum
    bottom_radius = 1.0  # Bottom radius of the frustum
    height = 1.5  # Height of the frustum
    num_segments = 36  # Number of segments to approximate the circle

    # Calculate angle increment for full circle division by segments
    angle_increment = 2 * np.pi / num_segments

    # Utilize structured arrays to define vertices and colors
    # Using numpy to manage data in a structured way
    vertices = np.zeros((num_segments * 2, 3), dtype=np.float32)
    colors = np.zeros((num_segments * 2, 4), dtype=np.float32)

    # Populate vertices and colors
    for i in range(num_segments):
        angle = i * angle_increment
        vertices[i] = [
            bottom_radius * np.cos(angle),
            bottom_radius * np.sin(angle),
            0,
        ]  # Bottom circle
        vertices[i + num_segments] = [
            top_radius * np.cos(angle),
            top_radius * np.sin(angle),
            height,
        ]  # Top circle
        colors[i] = [1, 0, 0, 1]  # Red color at the bottom
        colors[i + num_segments] = [0, 0, 1, 1]  # Blue color at the top

    # Add data to vertex and color writers using structured array data
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.zeros((num_segments * 2, 3), dtype=np.int32)
    for i in range(num_segments):
        triangle_indices[i] = [i, (i + 1) % num_segments, i + num_segments]
        triangle_indices[i + num_segments] = [
            (i + 1) % num_segments,
            i + num_segments,
            (i + 1) % num_segments + num_segments,
        ]

    # Create triangle primitives with static usage hint
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("spherical_frustum_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomTriangles,
)


def construct_torus_knot(p=2, q=3, num_vertices=100, radius=1.0):
    """
    Construct a torus knot geometry using a structured array approach for vertex and color data management.
    A torus knot is a type of knot that lies on the surface of a torus in three-dimensional space.
    The parameters 'p' and 'q' are integers that determine the type of torus knot.
    The 'num_vertices' parameter specifies the number of vertices to be used in constructing the knot.
    The 'radius' parameter specifies the scale of the torus knot.

    Parameters:
        p (int): Number of times the knot wraps around the torus tube.
        q (int): Number of times the knot goes around the torus center.
        num_vertices (int): Number of vertices to generate for the knot.
        radius (float): Scale factor for the size of the torus knot.

    Returns:
        GeomNode: A geometry node containing the constructed torus knot geometry.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()
    vertex_data = GeomVertexData(
        "torus_knot_vertices_and_colors", vertex_format, Geom.UHStatic
    )
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Initialize structured arrays for vertices and colors
    vertices = np.zeros((num_vertices, 3), dtype=np.float32)
    colors = np.zeros((num_vertices, 4), dtype=np.float32)

    # Calculate angle increment and torus dimensions
    angle_increment = 2 * np.pi / num_vertices
    tube_radius = 0.1 * radius  # Tube radius as a fraction of overall radius

    # Populate the vertices and colors arrays
    for i in range(num_vertices):
        phi = i * angle_increment * p
        theta = i * angle_increment * q
        x = (radius + tube_radius * np.cos(q * phi)) * np.cos(p * phi)
        y = (radius + tube_radius * np.cos(q * phi)) * np.sin(p * phi)
        z = tube_radius * np.sin(q * phi)
        vertices[i] = [x, y, z]
        colors[i] = [
            np.sin(phi),
            np.cos(phi),
            np.abs(np.sin(theta)),
            1.0,
        ]  # Color varies with phi and theta

    # Add data to vertex and color writers using structured array data
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            (i, (i + 1) % num_vertices, (i + 2) % num_vertices)
            for i in range(num_vertices)
        ],
        dtype=np.int32,
    )
    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("torus_knot_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_trefoil_knot():
    pass


def construct_mobius_strip():
    pass


def construct_klein_bottle():
    """
    Construct a Klein Bottle using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a Klein Bottle with detailed vertex and color definitions, using structured arrays for optimal data management.

    Returns:
    - GeomNode: a geometry node containing the constructed Klein Bottle geometry.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "klein_bottle_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Utilize structured arrays to define vertices and colors
    num_u_segments = 150  # Number of segments along the u-axis
    num_v_segments = 75  # Number of segments along the v-axis
    vertices = np.zeros((num_u_segments * num_v_segments, 3), dtype=np.float32)
    colors = np.zeros((num_u_segments * num_v_segments, 4), dtype=np.float32)

    # Populate vertices and colors using the parametric equations of a Klein Bottle
    for u_index in range(num_u_segments):
        u = 2 * np.pi * u_index / num_u_segments
        for v_index in range(num_v_segments):
            v = 2 * np.pi * v_index / num_v_segments
            x = (2.5 + 1.5 * np.cos(v)) * np.cos(u)
            y = (2.5 + 1.5 * np.cos(v)) * np.sin(u)
            z = 1.5 * np.sin(v) + np.cos(u) * (u < np.pi)
            index = u_index * num_v_segments + v_index
            vertices[index] = [x, y, z]
            colors[index] = [
                np.sin(u),
                np.cos(v),
                np.abs(np.sin(v)),
                1.0,
            ]  # Color varies with u and v

    # Add data to vertex and color writers using structured array data
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            (
                (
                    i + j * num_v_segments,
                    (i + 1) % num_v_segments + j * num_v_segments,
                    i + (j + 1) % num_u_segments * num_v_segments,
                )
                if j < num_u_segments - 1
                else (
                    i + j * num_v_segments,
                    (i + 1) % num_v_segments + j * num_v_segments,
                    (i + 1) % num_v_segments,
                )
            )
            for j in range(num_u_segments)
            for i in range(num_v_segments)
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    # Create geometry and add the primitive
    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)

    # Create a geometry node and add the geometry to it
    geometry_node = GeomNode("klein_bottle_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node


def construct_torus(
    num_segments=100, tube_segments=50, major_radius=1.0, minor_radius=0.3
):
    """
    Construct a torus using vertex data and geom nodes, adhering to the highest standards of data management and efficiency.
    This function meticulously constructs a torus with detailed vertex and color definitions, using structured arrays for optimal data management.

    Parameters:
    - num_segments: int, the number of segments along the major radius of the torus.
    - tube_segments: int, the number of segments along the minor radius of the torus.
    - major_radius: float, the major radius of the torus.
    - minor_radius: float, the minor radius of the torus.

    Returns:
    - GeomNode: a geometry node containing the constructed torus geometry.
    """
    # Define the vertex format with 3D coordinates and RGBA colors
    vertex_format = GeomVertexFormat.getV3c4()

    # Create vertex data container with static usage hint for efficiency
    vertex_data = GeomVertexData(
        "torus_vertices_and_colors", vertex_format, Geom.UHStatic
    )

    # Writers for vertices and colors
    vertex_writer = GeomVertexWriter(vertex_data, "vertex")
    color_writer = GeomVertexWriter(vertex_data, "color")

    # Utilize structured arrays to define vertices and colors
    vertices = np.zeros((num_segments * tube_segments, 3), dtype=np.float32)
    colors = np.zeros((num_segments * tube_segments, 4), dtype=np.float32)

    # Populate vertices and colors
    for i in range(num_segments):
        theta = 2 * np.pi * i / num_segments
        for j in range(tube_segments):
            phi = 2 * np.pi * j / tube_segments
            x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
            y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
            z = minor_radius * np.sin(phi)
            index = i * tube_segments + j
            vertices[index] = [x, y, z]
            colors[index] = [
                np.sin(theta),
                np.cos(phi),
                np.abs(np.sin(phi)),
                1.0,
            ]  # Color varies with theta and phi

    # Add data to vertex and color writers using structured array data
    for vertex, color in zip(vertices, colors):
        vertex_writer.addData3f(*vertex)
        color_writer.addData4f(*color)

    # Define triangles using indices and numpy arrays
    triangle_indices = np.array(
        [
            [
                i * tube_segments + j,
                (i * tube_segments + (j + 1) % tube_segments),
                ((i + 1) % num_segments) * tube_segments + j,
            ]
            for i in range(num_segments)
            for j in range(tube_segments)
        ]
        + [
            [
                (i * tube_segments + (j + 1) % tube_segments),
                ((i + 1) % num_segments) * tube_segments + (j + 1) % tube_segments,
                ((i + 1) % num_segments) * tube_segments + j,
            ]
            for i in range(num_segments)
            for j in range(tube_segments)
        ],
        dtype=np.int32,
    )

    triangles = GeomTriangles(Geom.UHStatic)
    for tri in triangle_indices:
        triangles.addVertices(*tri)

    geometry = Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = GeomNode("torus_geom_node")
    geometry_node.addGeom(geometry)

    return geometry_node
